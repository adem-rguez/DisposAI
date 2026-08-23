mod grpc;
mod http;
mod media_store;
mod orchestrator_tools;
mod profiler;
mod registry;
mod session;
mod task_tags;
mod tool_fallback;
mod vram;

use std::net::SocketAddr;
use std::sync::Arc;
use std::collections::HashMap;
use std::sync::atomic::AtomicU16;
use llama_backend::{process_tracker::ChildRegistry, LlamaBackend};
use sd_backend::SdBackend;
use mesh_backend::MeshBackend;
use whisper_backend::WhisperBackend;
use tts_backend::TtsBackend;
use video_backend::VideoBackend;
use moe_cache::MoeExpertCache;
use pool_protocol::{ClusterPoolManager, PeerNode};
use media_store::MediaStore;
use proto::daemon_service_server::DaemonServiceServer;
use registry::BackendRegistry;
use session::SessionManager;
use tracing::{info, warn};
use vram::VramArbiter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    info!("Starting DisposAI Local Inference Daemon v{}", env!("CARGO_PKG_VERSION"));

    // 0. Child-process tracker — shared by the legacy backend and the
    //    HTTP `/v1/model/load` path so spawned llama-server.exe processes
    //    can be terminated on shutdown or on targeted unload.
    let children = Arc::new(ChildRegistry::new());

    // 1. Probe Hardware & Initialize VRAM Arbiter
    let hardware_info = profiler::HardwareProfiler::probe();
    let vram_arbiter = Arc::new(VramArbiter::new(hardware_info.total_vram_bytes));

    // 2. Initialize Session Manager, MoE Cache, and Cluster Pool Manager
    let session_manager = Arc::new(SessionManager::new());
    let moe_cache = Arc::new(MoeExpertCache::new(4));
    let cluster_pool = Arc::new(ClusterPoolManager::new("local-node-primary".to_string()));

    // Register primary node
    cluster_pool
        .register_peer(PeerNode {
            node_id: "local-node-primary".to_string(),
            address: "127.0.0.1:50051".to_string(),
            free_vram_bytes: hardware_info.free_vram_bytes,
            total_vram_bytes: hardware_info.total_vram_bytes,
            latency_ms: 1,
        })
        .await;

    // 3. Initialize Backend Registry & Register plugins across all modalities
    let registry = Arc::new(BackendRegistry::new());

    registry.register_backend(Box::new(LlamaBackend::new(children.clone()))).await;
    info!("Registered backend plugin: llama.cpp (Text, Embedding)");

    registry.register_backend(Box::new(SdBackend::new())).await;
    info!("Registered backend plugin: stable-diffusion.cpp (Image)");

    registry.register_backend(Box::new(MeshBackend::new())).await;
    info!("Registered backend plugin: mesh3d-generator (Mesh3D)");

    registry.register_backend(Box::new(WhisperBackend::new())).await;
    info!("Registered backend plugin: whisper.cpp (Audio ASR)");

    registry.register_backend(Box::new(TtsBackend::new())).await;
    info!("Registered backend plugin: Kokoro TTS (Audio TTS)");

    registry.register_backend(Box::new(VideoBackend::new())).await;
    info!("Registered backend plugin: Wan Video Runner (Video)");

    // 3.5 Initialize Media Store
    let media_disk_dir = std::env::current_dir()?.join("generated-media");
    let media_store = Arc::new(MediaStore::new(1800, media_disk_dir)); // 30-minute TTL by default

    // Per-conversation registry of generated/uploaded media assets, so the orchestrator
    // can resolve pronoun references like "edit it" without vision input.
    let generated_assets: media_store::GeneratedAssetRegistry =
        Arc::new(tokio::sync::Mutex::new(HashMap::new()));

    // Periodic sweep of expired in-memory media (no-op when persisting to disk) and
    // idle conversation asset buckets.
    {
        let media_store = media_store.clone();
        let generated_assets = generated_assets.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
            loop {
                interval.tick().await;
                media_store.cleanup_expired().await;
                media_store::cleanup_expired_assets(&generated_assets).await;
            }
        });
    }

    // 4. Start HTTP OpenAI-compatible server & Dashboard on 0.0.0.0:8080
    let loaded_models = Arc::new(tokio::sync::Mutex::new(HashMap::<String, http::LoadedModelEntry>::new()));
    let next_port = Arc::new(AtomicU16::new(50052));
    let active_model = Arc::new(tokio::sync::Mutex::new(http::LoadedModelStatus {
        is_loaded: false,
        model_path: None,
        gpu_layers: None,
        context_size: None,
    }));

    // Shared shutdown signal: ctrl_c fires once and wakes both servers.
    // broadcast::channel buffers the signal so a Ctrl+C that arrives before
    // either server's shutdown future is polled is still delivered. Also used
    // by the /shutdown HTTP endpoint for graceful shutdown from the launcher.
    let (shutdown_tx, _) = tokio::sync::broadcast::channel::<()>(1);

    let http_state = http::AppState {
        registry: registry.clone(),
        session_manager: session_manager.clone(),
        moe_cache: moe_cache.clone(),
        cluster_pool: cluster_pool.clone(),
        loaded_models,
        next_port,
        active_model,
        children: children.clone(),
        shutdown_signal: shutdown_tx.clone(),
        media_store,
        hf_downloads: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        hf_cancel: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        job_progress: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        job_cancel: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        studio_models: Arc::new(tokio::sync::Mutex::new(std::collections::HashSet::new())),
        hf_token: Arc::new(tokio::sync::Mutex::new(None)),
        generated_assets: generated_assets.clone(),
    };
    let app = http::create_router(http_state);
    let http_addr: SocketAddr = "0.0.0.0:8080".parse()?;
    let rx_axum = shutdown_tx.subscribe();
    let rx_grpc = shutdown_tx.subscribe();

    let shutdown_tx_ctrlc = shutdown_tx.clone();
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            info!("Ctrl+C received; initiating graceful shutdown");
            let _ = shutdown_tx_ctrlc.send(());
        } else {
            warn!("ctrl_c listener exited without a signal");
        }
    });

    // Also listen for SIGTERM (Unix). The Electron launcher's
    // `daemon.kill()` on app exit sends SIGTERM on Unix; without this
    // listener, the daemon would be terminated without running its
    // graceful shutdown path — which means `drop(children)` at the end
    // of main never fires and the spawned llama-server.exe children are
    // orphaned. On Windows there's no SIGTERM, so this block is gated.
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        match signal(SignalKind::terminate()) {
            Ok(mut sigterm) => {
                let shutdown_tx_term = shutdown_tx.clone();
                tokio::spawn(async move {
                    if sigterm.recv().await.is_some() {
                        info!("SIGTERM received; initiating graceful shutdown");
                        let _ = shutdown_tx_term.send(());
                    }
                });
            }
            Err(e) => warn!("failed to install SIGTERM handler: {}", e),
        }
    }

    // HTTP server runs in a detached spawn (original layout) and now also
    // responds to ctrl_c via with_graceful_shutdown.
    let axum_task = tokio::spawn(async move {
        info!("HTTP REST server & Web Dashboard listening on http://{}", http_addr);
        let listener = match tokio::net::TcpListener::bind(http_addr).await {
            Ok(l) => l,
            Err(e) => {
                warn!("HTTP bind failed: {}", e);
                return;
            }
        };
        let mut rx = rx_axum;
        if let Err(e) = axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = rx.recv().await;
            })
            .await
        {
            warn!("HTTP server stopped with error: {}", e);
        }
    });

    // 5. Start gRPC Daemon Service on 0.0.0.0:50051 — runs in foreground
    //    (matches original layout) and now also responds to ctrl_c.
    let grpc_addr: SocketAddr = "0.0.0.0:50051".parse()?;
    let grpc_service = grpc::DaemonGrpcService::new(registry.clone(), vram_arbiter.clone());
    info!("gRPC Daemon server listening on {}", grpc_addr);
    let mut rx = rx_grpc;
    tonic::transport::Server::builder()
        .add_service(DaemonServiceServer::new(grpc_service))
        .serve_with_shutdown(grpc_addr, async move {
            let _ = rx.recv().await;
        })
        .await?;

    // gRPC ended (either errored or via shutdown). Let axum finish too.
    let _ = axum_task.await;

    // Explicit drop: the axum task has already dropped its AppState (and the
    // BackendRegistry within it), so this is the last outstanding
    // Arc<ChildRegistry> clone. Dropping it runs the registry's Drop impl,
    // which kills every still-tracked llama-server child before main returns.
    drop(children);

    Ok(())
}