use std::sync::Arc;
use proto::daemon_service_server::DaemonService;
use proto::{
    GenerateRequest, GenerateResponse, LoadModelRequest, LoadModelResponse, StatusRequest,
    StatusResponse, StreamChunk, UnloadModelRequest, UnloadModelResponse,
};
use tonic::{Request, Response, Status};
use crate::registry::BackendRegistry;
use crate::vram::VramArbiter;
use backend_trait::{InferenceRequest, Modality, SamplingParams};

pub struct DaemonGrpcService {
    registry: Arc<BackendRegistry>,
    vram_arbiter: Arc<VramArbiter>,
}

impl DaemonGrpcService {
    pub fn new(registry: Arc<BackendRegistry>, vram_arbiter: Arc<VramArbiter>) -> Self {
        Self {
            registry,
            vram_arbiter,
        }
    }
}

#[tonic::async_trait]
impl DaemonService for DaemonGrpcService {
    type GenerateStreamStream = tokio_stream::wrappers::ReceiverStream<Result<StreamChunk, Status>>;

    async fn generate(
        &self,
        request: Request<GenerateRequest>,
    ) -> Result<Response<GenerateResponse>, Status> {
        let req = request.into_inner();
        let backend_arc = self
            .registry
            .get_backend(Modality::Text)
            .await
            .ok_or_else(|| Status::unavailable("No backend available for modality"))?;

        let sampling = req.sampling.unwrap_or_default();
        let inf_req = InferenceRequest {
            request_id: req.request_id.clone(),
            prompt: req.prompt,
            messages: None,
            sampling: SamplingParams {
                temperature: sampling.temperature,
                top_p: sampling.top_p,
                top_k: sampling.top_k,
                max_tokens: sampling.max_tokens,
                stop_sequences: sampling.stop_sequences,
            },
            modality: Modality::Text,
            image_input: if req.image_input.is_empty() {
                None
            } else {
                Some(req.image_input)
            },
            tools: None,
            tool_choice: None,
            image_params: None,
            mesh_params: None,
            audio_params: None,
            video_params: None,
            cancel: None,
        };

        let backend = backend_arc.read().await;
        let res = backend
            .generate(inf_req)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(GenerateResponse {
            request_id: res.request_id,
            output_text: res.output_text,
            output_data: res.output_data.unwrap_or_default(),
            tokens_generated: res.tokens_generated,
            generation_time_ms: res.generation_time_ms,
        }))
    }

    async fn generate_stream(
        &self,
        _request: Request<GenerateRequest>,
    ) -> Result<Response<Self::GenerateStreamStream>, Status> {
        Err(Status::unimplemented("Stream generation via gRPC coming in Phase 1"))
    }

    async fn load_model(
        &self,
        request: Request<LoadModelRequest>,
    ) -> Result<Response<LoadModelResponse>, Status> {
        let req = request.into_inner();
        let backend_arc = self
            .registry
            .get_backend(Modality::Text)
            .await
            .ok_or_else(|| Status::unavailable("No backend available"))?;

        let mut backend = backend_arc.write().await;
        let path = std::path::Path::new(&req.model_path);
        
        let load_opts = backend_trait::LoadOptions {
            gpu_layers: req.gpu_layers,
            context_size: req.context_size,
            mmproj_path: req.mmproj_path.clone(),
            ..Default::default()
        };

        let estimate = backend
            .estimate_vram(path, &load_opts)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        backend
            .load_model(path, &load_opts)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        self.vram_arbiter
            .register_allocation(
                req.model_path.clone(),
                backend.name().to_string(),
                estimate.required_bytes,
            )
            .await;

        Ok(Response::new(LoadModelResponse {
            success: true,
            message: "Model successfully loaded".to_string(),
            estimated_vram_bytes: estimate.required_bytes,
        }))
    }

    async fn unload_model(
        &self,
        request: Request<UnloadModelRequest>,
    ) -> Result<Response<UnloadModelResponse>, Status> {
        let req = request.into_inner();
        let backend_arc = self
            .registry
            .get_backend(Modality::Text)
            .await
            .ok_or_else(|| Status::unavailable("No backend available"))?;

        let mut backend = backend_arc.write().await;
        backend
            .unload_model()
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        self.vram_arbiter.remove_allocation(&req.model_id).await;

        Ok(Response::new(UnloadModelResponse { success: true }))
    }

    async fn get_status(
        &self,
        _request: Request<StatusRequest>,
    ) -> Result<Response<StatusResponse>, Status> {
        let free_vram = self.vram_arbiter.get_free_bytes().await;
        let allocated_vram = self.vram_arbiter.get_allocated_bytes().await;
        let total_vram = free_vram + allocated_vram;

        Ok(Response::new(StatusResponse {
            version: env!("CARGO_PKG_VERSION").to_string(),
            total_vram_bytes: total_vram,
            free_vram_bytes: free_vram,
            backends: vec![],
        }))
    }
}
