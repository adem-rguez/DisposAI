import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  MessageSquare, Sparkles, Volume2, Video,
  Cpu, HardDrive, Zap, Send, Play, Image, FileAudio, RefreshCw,
  Brain, ChevronDown, ChevronRight, ChevronLeft, Sliders, Folder, Power, Layers, Settings,
  CheckCircle2, XCircle, PackagePlus, Box, Boxes, Paperclip, X, Pencil,
  Search, Download, Globe, Loader, Check, Heart, Wand2, ArrowUpRight, Trash2, Square, Pause, Plus, Copy
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import Mesh3DViewer, { isMeshResource, guessMeshFormat } from './Mesh3DViewer';
import { ErrorLogProvider, ErrorToasts, useErrorLog } from './ErrorLog';
import disposLogo from './assets/dispos_logo.png';
import './App.css';

async function browseForFile(defaultPath, filters, properties) {
  const { ipcRenderer } = window.require('electron');
  return ipcRenderer.invoke('select-file', { defaultPath, filters, properties });
}

function findResponseStart(text) {
  const patterns = [
    /\n\n(?=Hello|Hi|Hey|Greetings|Sure|Certainly|I'm|I am|Here|As an AI)/i,
    /\n\n(?="[A-Z])/i
  ];
  for (const pat of patterns) {
    const match = text.match(pat);
    if (match && match.index > 50) {
      return match.index;
    }
  }
  return -1;
}

function parseThinking(content) {
  if (!content) return { thinking: '', answer: '', hadThinkTag: false };
  const str = content.trim();

  // 1. Check for explicit <think> tag
  const thinkStart = str.indexOf('<think>');
  const thinkEnd = str.indexOf('</think>');

  if (thinkStart !== -1) {
    if (thinkEnd !== -1 && thinkEnd > thinkStart) {
      const thinking = str.substring(thinkStart + 7, thinkEnd).trim();
      const answer = (str.substring(0, thinkStart) + '\n' + str.substring(thinkEnd + 8)).trim();
      return { thinking, answer, hadThinkTag: true };
    } else {
      // <think> opened but never closed — entire remainder is thinking, no answer yet
      const thinking = str.substring(thinkStart + 7).trim();
      const answer = str.substring(0, thinkStart).trim();
      return { thinking, answer, hadThinkTag: true };
    }
  }

  // 2. Check for </think> tag without opening <think>
  if (thinkEnd !== -1) {
    const thinking = str.substring(0, thinkEnd).replace(/^(?:Thinking|Thought)\s+Process:?\s*/i, '').trim();
    const answer = str.substring(thinkEnd + 8).trim();
    return { thinking, answer, hadThinkTag: true };
  }

  // 3. Check for "Thought Process" or "Thinking Process" block without <think> tags
  const tpRegex = /^(?:Thought|Thinking)\s+Process(?:\s*\(\d+\s*words\))?:?\s*/i;
  if (tpRegex.test(str)) {
    const body = str.replace(tpRegex, '').trim();
    const responseStart = findResponseStart(body);
    if (responseStart !== -1) {
      const thinking = body.substring(0, responseStart).trim();
      const answer = body.substring(responseStart).trim();
      return { thinking, answer, hadThinkTag: true };
    }
    return { thinking: body, answer: '', hadThinkTag: true };
  }

  // 4. Default: No reasoning detected, whole text is answer
  return { thinking: '', answer: str, hadThinkTag: false };
}

function ThinkingBlock({ thinkText, hadThinkTag }) {
  const [isOpen, setIsOpen] = useState(false);

  // Only render when there is actual reasoning content
  if (!hadThinkTag || !thinkText || thinkText.trim().length === 0) return null;

  const wordCount = thinkText.trim().split(/\s+/).filter(Boolean).length;

  return (
    <div className="thinking-accordion">
      <button className="thinking-header" onClick={() => setIsOpen(!isOpen)}>
        <div className="thinking-title">
          <Brain size={14} className="thinking-icon" />
          <span>Thought Process ({wordCount} words)</span>
        </div>
        {isOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
      </button>
      {isOpen && (
        <div className="thinking-content">
          {thinkText}
        </div>
      )}
    </div>
  );
}

// A single "Used tool: X" chip. Expands (same accordion pattern as
// ThinkingBlock) to show the request details for that specific call —
// model name and full arguments, when the backend has sent them.
function ToolCallChip({ event, onCancel, onExpand }) {
  const [isOpen, setIsOpen] = useState(false);
  const modelName = event.arguments?.model;
  const label = event.status === 'executing' ? `Calling ${event.name}...`
    : event.status === 'done' ? `Used tool: ${event.name}${modelName ? ` (${modelName})` : ''}`
      : event.status === 'error' ? `Tool ${event.name} failed: ${event.detail}`
        : event.status === 'cancelled' ? `Generation cancelled: ${event.name}`
          : '';

  return (
    <div className={`tool-status-indicator ${event.status}`} style={{ flexDirection: 'column', alignItems: 'stretch' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <button className="thinking-header" style={{ padding: 0, background: 'transparent', color: 'inherit', flex: 1 }} onClick={() => setIsOpen(!isOpen)}>
          <div className="thinking-title">
            {event.status === 'executing' && <Loader size={14} className="spin" />}
            <span>{label}</span>
          </div>
          {isOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </button>
        {event.name === 'run_model' && event.jobId && (
          <button className="tool-status-cancel-btn" onClick={onExpand} title="Open in Studio">
            <ArrowUpRight size={14} /> Expand to Studio
          </button>
        )}
        {event.status === 'executing' && event.jobId && (
          <button className="tool-status-cancel-btn" onClick={onCancel}>
            Cancel
          </button>
        )}
      </div>
      {event.progress && <GenProgressBar progress={event.progress} />}
      {isOpen && (
        <div className="thinking-content">
          {event.arguments
            ? JSON.stringify(event.arguments, null, 2)
            : 'No request details available for this call (the daemon does not currently send tool call arguments to the UI).'}
        </div>
      )}
    </div>
  );
}

// Generation progress bar, shared by the image/mesh/tts studios. `progress` is
// a GenerationProgress record (see /v1/jobs/:id/progress) or null when idle.
function GenProgressBar({ progress }) {
  if (!progress) return null;
  const determinate = progress.percent >= 0 && progress.total > 0;
  return (
    <div className="gen-progress">
      {determinate ? (
        <>
          <div className="gen-progress-bar">
            <div className="gen-progress-fill" style={{ width: `${Math.min(100, progress.percent)}%` }} />
          </div>
          <span className="gen-progress-text">
            {progress.step}/{progress.total} · {Math.round(progress.percent)}% · {progress.phase}
          </span>
        </>
      ) : (
        <>
          <div className="gen-progress-bar gen-progress-indeterminate">
            <div className="gen-progress-fill-indeterminate" />
          </div>
          <span className="gen-progress-text">{progress.phase}</span>
        </>
      )}
    </div>
  );
}

// Stable identity for a loaded model. Backend model_ids are timestamped
// (mdl-{millis}-{port}), so reloading the same file yields a new id. The file's
// basename is the only thing that survives across loads; normalize it so
// backslash vs forward slash, casing, and quantization-tag drift all collapse
// to the same key.
function canonicalStudioKey(modelPath, modelName) {
  const raw = (modelPath || modelName || '').toString();
  const basename = raw.split(/[\\/]/).pop() || raw;
  const stripped = basename.replace(/\.(gguf|safetensors|onnx|bin|ckpt)$/i, '');
  const cleaned = stripped.toLowerCase().replace(/\s+/g, '-').trim();
  return cleaned || raw;
}

// A chat session's "last modified" time: bumped on every message write
// (see syncMessages), falling back to createdAt for sessions that haven't
// had one yet.
function chatLastModified(chat) {
  return chat.updatedAt ?? chat.createdAt ?? 0;
}

// Mirrors `select_adapter()` / `ADAPTER_REGISTRY` in scripts/threed_server.py:
// same match-token precedence (first match wins, "cube-placeholder" is the
// fallback), kept in sync manually since the Python side is the source of
// truth for which adapter actually loads. Used to key into the per-adapter
// param schema from `GET /v1/models3d/schema`.
const MESH3D_ADAPTER_MATCH_ORDER = [
  { id: 'llama-mesh', tokens: ['llama-mesh', 'llama_mesh', 'llamamesh'] },
  { id: 'point-e', tokens: ['point-e', 'point_e', 'pointe'] },
  { id: 'shap-e', tokens: ['shap-e', 'shap_e', 'shape'] },
  { id: 'triposr', tokens: ['triposr', 'tripo'] },
  { id: 'stable-fast-3d', tokens: ['sf3d', 'stable-fast-3d', 'stable_fast_3d'] },
  { id: 'vggt', tokens: ['vggt', 'vggt-1b', 'vggt_1b'] },
  { id: 'cube-placeholder', tokens: ['instantmesh', 'instant-mesh', 'instant_mesh'] },
  { id: 'trellis', tokens: ['trellis'] },
  { id: 'hunyuan3d', tokens: ['hunyuan3d', 'hunyuan-3d', 'hunyuan_3d'] },
  { id: 'cube-placeholder', tokens: ['wonder3d', 'sv3d', 'zero123', 'zero-1-2-3'] },
  { id: 'cube-placeholder', tokens: ['crm', 'lgm'] },
];

function detectMesh3dAdapterId(modelPath) {
  if (!modelPath) return 'cube-placeholder';
  const parts = modelPath.replace(/\\/g, '/').split('/').filter(Boolean);
  const combined = `${parts[parts.length - 2] || ''} ${parts[parts.length - 1] || ''}`.toLowerCase();
  const match = MESH3D_ADAPTER_MATCH_ORDER.find(({ tokens }) => tokens.some(token => combined.includes(token)));
  return match ? match.id : 'cube-placeholder';
}

// Initial control value for a schema param. Respects `default` when present;
// a null/undefined default (e.g. shap-e's `seed`) is left as `null` for
// numeric types so it's sent through as-is and the backend's own PARAM_SPEC
// default applies (see `resolve_params()` in scripts/threed_server.py).
function meshParamDefaultValue(p) {
  if (p.default !== null && p.default !== undefined) return p.default;
  if (p.type === 'bool') return false;
  if (p.type === 'str') return '';
  if (p.type === 'enum') return p.choices?.[0] ?? '';
  return null;
}

// Generic control for a single `GET /v1/.../schema`-style param (used by both
// the mesh3d and video panels, which share the same {name, type, ...} shape).
function SchemaParamField({ param: p, value, onChange }) {
  if (p.type === 'bool') {
    return (
      <div className="slider-header" style={{ marginBottom: '1rem' }}>
        <div><strong>{p.name}</strong></div>
        <input
          type="checkbox"
          checked={Boolean(value)}
          onChange={e => onChange(e.target.checked)}
        />
      </div>
    );
  }
  let control;
  if (p.type === 'enum') {
    control = (
      <select value={value ?? ''} onChange={e => onChange(e.target.value)}>
        {(p.choices || []).map(choice => <option key={choice} value={choice}>{choice}</option>)}
      </select>
    );
  } else if (p.type === 'int' || p.type === 'float') {
    control = (
      <input
        type="number"
        step={p.type === 'float' ? '0.1' : '1'}
        min={p.min ?? undefined}
        max={p.max ?? undefined}
        value={value ?? ''}
        onChange={e => onChange(e.target.value === '' ? null : Number(e.target.value))}
      />
    );
  } else {
    control = <input type="text" value={value ?? ''} onChange={e => onChange(e.target.value)} />;
  }
  return <div className="form-group"><label>{p.name}</label>{control}</div>;
}

function formatVideoTime(seconds) {
  if (!isFinite(seconds) || seconds < 0) return '0:00';
  const total = Math.floor(seconds);
  const mins = Math.floor(total / 60);
  const secs = total % 60;
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

function VideoPlayer({ src, autoPlay, style, className }) {
  const videoRef = useRef(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    setPlaying(false);
    setCurrentTime(0);
    setDuration(0);
  }, [src]);

  const togglePlay = () => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      video.play();
    } else {
      video.pause();
    }
  };

  const handleSeek = (e) => {
    const video = videoRef.current;
    if (!video || !duration) return;
    video.currentTime = Number(e.target.value);
    setCurrentTime(Number(e.target.value));
  };

  return (
    <div className={`video-player${className ? ` ${className}` : ''}`} style={style}>
      <video
        ref={videoRef}
        src={src}
        autoPlay={autoPlay}
        className="video-player-el"
        onClick={togglePlay}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onLoadedMetadata={(e) => setDuration(e.currentTarget.duration || 0)}
        onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
        onEnded={() => setPlaying(false)}
      />
      <div className="video-player-controls">
        <button type="button" className="video-player-playbtn" onClick={togglePlay}>
          {playing ? <Pause size={16} /> : <Play size={16} />}
        </button>
        <input
          type="range"
          className="video-player-seek"
          min={0}
          max={duration || 0}
          step={0.01}
          value={currentTime}
          onChange={handleSeek}
        />
        <span className="video-player-time">{formatVideoTime(currentTime)} / {formatVideoTime(duration)}</span>
      </div>
    </div>
  );
}

function AttachmentPreview({ attachment }) {
  if (attachment.type.startsWith('image/')) {
    return <img className="chat-attachment-image" src={attachment.dataUrl} alt={attachment.name} />;
  }
  if (attachment.type.startsWith('audio/')) {
    return <audio className="chat-attachment-media" controls src={attachment.dataUrl} />;
  }
  if (attachment.type.startsWith('video/')) {
    return <video className="chat-attachment-media" controls src={attachment.dataUrl} />;
  }
  if (isMeshResource(attachment.type, attachment.name)) {
    return <Mesh3DViewer base64={attachment.dataUrl} format={guessMeshFormat(attachment.name)} />;
  }
  return <a className="chat-attachment-file" href={attachment.dataUrl} download={attachment.name}>{attachment.name}</a>;
}

function CopyMessageButton({ text }) {
  const [copied, setCopied] = useState(false);
  const handleCopy = () => {
    navigator.clipboard.writeText(text || '');
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };
  return (
    <button type="button" className="copy-msg-btn" onClick={handleCopy} title={copied ? 'Copied!' : 'Copy message'}>
      {copied ? <Check size={13} /> : <Copy size={13} />}
    </button>
  );
}

// Icon shown for a studio row when the sidebar is folded. Reuses the same
// per-author avatar endpoint/cache as the HF catalog (see hf-author-avatar
// usage in the Discover tab below); the author is guessed from the model
// path. Local downloads are NOT stored under a nested author/repo tree —
// sanitize_repo_id() in http.rs flattens "org/repo" into a single folder
// named "<author>_<repo>" (e.g. models/cstr_kokoro-82m-GGUF/file.gguf, or
// models/VAST-AI_TripoSG/ for a collapsed HF model directory per
// hf_model_dir_entry). So the author is the part of that one folder name
// before the first underscore, not a separate path segment.
// Falls back to a plain Box icon when there's no plausible author segment
// or the avatar fails to load (unknown author, no network, etc).
function StudioAvatarIcon({ modelPath, size = 26 }) {
  const [failed, setFailed] = useState(false);
  const parts = (modelPath || '').replace(/\\/g, '/').split('/').filter(Boolean);
  const isFile = /\.[A-Za-z0-9]+$/.test(parts[parts.length - 1] || '');
  const folderName = isFile ? parts[parts.length - 2] : parts[parts.length - 1];
  const author = folderName ? folderName.split('_')[0] : null;
  if (!author || failed) return <Box size={size} />;
  return (
    <img
      className="hf-author-avatar sidebar-fold-avatar"
      style={{ width: size, height: size }}
      src={`http://127.0.0.1:8080/v1/model/hf-avatar?author=${encodeURIComponent(author)}&v=2`}
      alt=""
      onError={() => setFailed(true)}
    />
  );
}

// Renders the model-status-strip indicator for one of 4 states: 'online'
// (green, loaded & running), 'loading' (amber, load request in-flight),
// 'idle' (red dot, selected for this studio but not loaded), or 'missing'
// (red X, the selected model no longer exists in the catalog). A null state
// (no model associated at all) uses the neutral base dot — amber is reserved
// for 'loading' so the two can't be confused.
function StatusDot({ state }) {
  if (state === 'missing') return <div className="status-dot-lg missing"><XCircle size={12} /></div>;
  return <div className={state ? `status-dot-lg ${state}` : 'status-dot-lg'}></div>;
}

// Mirrors crates/daemon-core/src/task_tags.rs::backend_category_for_tags priority
// order (AudioTts > AudioAsr > Image > Video > Mesh3D > chat/llama-server path),
// reimplemented here since there's no shared code between the daemon and this
// frontend. Category keys line up with the studio tab ids used below.
const TASK_TAG_CATEGORY = {
  'text-to-speech': 'tts',
  'automatic-speech-recognition': 'audio',
  'audio-classification': 'audio',
  'text-to-image': 'image',
  'image-to-image': 'image',
  'unconditional-image-generation': 'image',
  'image-classification': 'image',
  'image-segmentation': 'image',
  'zero-shot-image-classification': 'image',
  'text-to-video': 'video',
  'image-to-video': 'video',
  'video-to-video': 'video',
  'video-classification': 'video',
  'video-text-to-text': 'video',
  'image-text-to-video': 'video',
  'text-to-3d': 'mesh3d',
  'image-to-3d': 'mesh3d',
  'feature-extraction': 'embeddings',
  'text-generation': 'chat',
  'text2text-generation': 'chat',
  'conversational': 'chat',
  'image-text-to-text': 'chat',
  'visual-question-answering': 'chat',
};
const CATEGORY_PRIORITY = ['tts', 'audio', 'image', 'video', 'mesh3d', 'embeddings', 'chat'];
const CATEGORY_DEFAULT_TITLE = { tts: 'New Audio', audio: 'New Audio', image: 'New Image', video: 'New Video', mesh3d: 'New 3D Model', embeddings: 'New Embedding', chat: 'New chat' };
function categoryForTags(tags) {
  const set = new Set((tags || []).map(t => TASK_TAG_CATEGORY[t]).filter(Boolean));
  for (const c of CATEGORY_PRIORITY) if (set.has(c)) return c;
  return 'chat';
}

const HF_MODEL_TYPE_GROUPS = [
  {
    group: 'Multimodal', types: [
      { label: 'Audio-Text-to-Text', tag: 'audio-text-to-text' },
      { label: 'Image-Text-to-Text', tag: 'image-text-to-text' },
      { label: 'Visual Question Answering', tag: 'visual-question-answering' },
      { label: 'Document Question Answering', tag: 'document-question-answering' },
      { label: 'Video-Text-to-Text', tag: 'video-text-to-text' },
      { label: 'Visual Document Retrieval', tag: 'visual-document-retrieval' },
      { label: 'Any-to-Any', tag: 'any-to-any' },
    ]
  },
  {
    group: 'Computer Vision', types: [
      { label: 'Depth Estimation', tag: 'depth-estimation' },
      { label: 'Image Classification', tag: 'image-classification' },
      { label: 'Object Detection', tag: 'object-detection' },
      { label: 'Image Segmentation', tag: 'image-segmentation' },
      { label: 'Text-to-Image', tag: 'text-to-image' },
      { label: 'Image-to-Text', tag: 'image-to-text' },
      { label: 'Image-to-Image', tag: 'image-to-image' },
      { label: 'Image-to-Video', tag: 'image-to-video' },
      { label: 'Unconditional Image Generation', tag: 'unconditional-image-generation' },
      { label: 'Video Classification', tag: 'video-classification' },
      { label: 'Text-to-Video', tag: 'text-to-video' },
      { label: 'Zero-Shot Image Classification', tag: 'zero-shot-image-classification' },
      { label: 'Mask Generation', tag: 'mask-generation' },
      { label: 'Zero-Shot Object Detection', tag: 'zero-shot-object-detection' },
      { label: 'Text-to-3D', tag: 'text-to-3d' },
      { label: 'Image-to-3D', tag: 'image-to-3d' },
      { label: 'Image Feature Extraction', tag: 'image-feature-extraction' },
      { label: 'Keypoint Detection', tag: 'keypoint-detection' },
    ]
  },
  {
    group: 'Natural Language Processing', types: [
      { label: 'Text Classification', tag: 'text-classification' },
      { label: 'Token Classification', tag: 'token-classification' },
      { label: 'Table Question Answering', tag: 'table-question-answering' },
      { label: 'Question Answering', tag: 'question-answering' },
      { label: 'Zero-Shot Classification', tag: 'zero-shot-classification' },
      { label: 'Translation', tag: 'translation' },
      { label: 'Summarization', tag: 'summarization' },
      { label: 'Feature Extraction', tag: 'feature-extraction' },
      { label: 'Text Generation', tag: 'text-generation' },
      { label: 'Text-to-Text Generation', tag: 'text2text-generation' },
      { label: 'Fill-Mask', tag: 'fill-mask' },
      { label: 'Sentence Similarity', tag: 'sentence-similarity' },
    ]
  },
  {
    group: 'Audio', types: [
      { label: 'Text-to-Speech', tag: 'text-to-speech' },
      { label: 'Text-to-Audio', tag: 'text-to-audio' },
      { label: 'Automatic Speech Recognition', tag: 'automatic-speech-recognition' },
      { label: 'Audio-to-Audio', tag: 'audio-to-audio' },
      { label: 'Audio Classification', tag: 'audio-classification' },
      { label: 'Voice Activity Detection', tag: 'voice-activity-detection' },
    ]
  },
  {
    group: 'Tabular', types: [
      { label: 'Tabular Classification', tag: 'tabular-classification' },
      { label: 'Tabular Regression', tag: 'tabular-regression' },
      { label: 'Time Series Forecasting', tag: 'time-series-forecasting' },
    ]
  },
  {
    group: 'Reinforcement Learning', types: [
      { label: 'Reinforcement Learning', tag: 'reinforcement-learning' },
      { label: 'Robotics', tag: 'robotics' },
    ]
  },
  {
    group: 'Other', types: [
      { label: 'Graph Machine Learning', tag: 'graph-machine-learning' },
    ]
  },
];

const HF_LANGUAGES = [
  { label: 'English', tag: 'en' },
  { label: 'Chinese', tag: 'zh' },
  { label: 'French', tag: 'fr' },
  { label: 'German', tag: 'de' },
  { label: 'Spanish', tag: 'es' },
  { label: 'Japanese', tag: 'ja' },
  { label: 'Korean', tag: 'ko' },
  { label: 'Russian', tag: 'ru' },
  { label: 'Arabic', tag: 'ar' },
  { label: 'Portuguese', tag: 'pt' },
];

const HF_PRECISIONS = [
  'Q2_K', 'Q3_K_M', 'Q4_0', 'Q4_K_M', 'Q5_K_M', 'Q6_K', 'Q8_0', 'F16',
].map(tag => ({ label: tag, tag }));

const HF_FORMATS = [
  { label: 'GGUF', tag: 'gguf' },
  { label: 'Safetensors', tag: 'safetensors' },
];

// Non-linear parameter-count stops (in billions). Top stop = "no upper limit".
const HF_PARAM_STOPS = [0, 0.5, 1, 3, 7, 13, 30, 70, 150, 500];

const HF_FILE_ROLE_INFO = {
  mmproj: { icon: '👁', tooltip: 'Gives the model vision — lets it understand images' },
  mtp: { icon: '⚡', tooltip: 'Speeds up generation — predicts several tokens ahead' },
  config: { icon: '⚙', tooltip: 'Model configuration' },
  tokenizer: { icon: '🔤', tooltip: 'Tokenizer / vocabulary' },
  shard: { icon: '🧩', tooltip: 'One part of the split model weights (needs all parts)' },
  index: { icon: '🗺', tooltip: 'Map of which weights live in which shard' },
  weights: { icon: '📦', tooltip: 'Model weights' },
  vae: { icon: '🎨', tooltip: 'VAE — encodes/decodes images' },
  unet: { icon: '🌀', tooltip: 'U-Net — the diffusion core' },
  text_encoder: { icon: '📝', tooltip: 'Text encoder — understands the prompt' },
};

const getHfFileRoleInfo = (file) => {
  const info = HF_FILE_ROLE_INFO[file?.role] || { icon: '📄', tooltip: 'Supporting file' };
  if (file?.role === 'weights' && file?.quant) {
    return { ...info, tooltip: `${info.tooltip} · ${file.quant}` };
  }
  return info;
};

const formatParamStop = (billions) => {
  if (billions <= 0) return '0';
  if (billions < 1) return `${Math.round(billions * 1000)}M`;
  return `${billions % 1 === 0 ? billions : billions.toFixed(1)}B`;
};

function AppInner() {
  const { pushError } = useErrorLog();
  const [activeTab, setActiveTab] = useState('models');
  const [modelPath, setModelPath] = useState(
    'models\\Qwen3.5-0.8B-Q8_0.gguf'
  );
  // The orchestrator (activeChatId === null) is a fixed, standalone
  // conversation — never a chatSessions entry, never the target of
  // startStudio/expandToolCallToStudio. Its transcript lives here, separate
  // from the per-session transcripts in chatSessions. Without this split, any
  // studio whose category resolves to 'chat' (or falls back to 'chat' when a
  // model's task_tags don't match a known category) rendered through the exact
  // same panel as the orchestrator, so opening it clobbered the orchestrator's
  // own view immediately — not just after leaving and coming back.
  const [orchestratorMessages, setOrchestratorMessages] = useState([
    {
      id: 'welcome',
      role: 'assistant',
      content: 'Hello! I am your DisposAI Local Inference engine running natively with NVIDIA CUDA GPU acceleration.\n\nEnter your GGUF model path and prompt below to generate real responses at over 140+ tokens/sec!',
      telemetry: null
    }
  ]);
  const [inputPrompt, setInputPrompt] = useState('');
  const [attachments, setAttachments] = useState([]);
  const [pendingModelChoice, setPendingModelChoice] = useState(null);
  const [editingMsgId, setEditingMsgId] = useState(null);
  const attachmentInputRef = useRef(null);
  const composerTextareaRef = useRef(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const defaultAppSettings = {
    autoSelectNewest: true,
    refreshSeconds: 3,
    showThoughtProcess: true,
    autopilot: false,
    mediaRetention: { ttl_seconds: 1800, persist_disk: false },
  };
  const [appSettings, setAppSettings] = useState(() => {
    try {
      const stored = JSON.parse(localStorage.getItem('dispos.general-settings'));
      return stored ? { ...defaultAppSettings, ...stored } : defaultAppSettings;
    } catch {
      return defaultAppSettings;
    }
  });

  useEffect(() => {
    localStorage.setItem('dispos.general-settings', JSON.stringify(appSettings));
  }, [appSettings]);

  // Sync saved media retention preference to daemon on startup (daemon defaults to 30min otherwise).
  useEffect(() => {
    fetch('http://127.0.0.1:8080/v1/config/media-retention', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(appSettings.mediaRetention),
    }).catch(() => { });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const [autopilot, setAutopilot] = useState(appSettings.autopilot ?? false);
  useEffect(() => { setAutopilot(appSettings.autopilot ?? false); }, [appSettings.autopilot]);
  const [reasoningEnabled, setReasoningEnabled] = useState(true);

  // Hardware stats state
  const [sysInfo, setSysInfo] = useState({
    cpu_cores: 20,
    total_ram_gb: 32.0,
    free_ram_gb: 20.0,
    total_vram_gb: 16.0,
    free_vram_gb: 12.0,
    gpu_name: null
  });

  // Image Studio State
  const [imgPrompt, setImgPrompt] = useState('A futuristic cyberpunk city skyline at sunset, photorealistic, 8k');
  const [imgNegativePrompt, setImgNegativePrompt] = useState('');
  const [imgSteps, setImgSteps] = useState(25);
  const [imgCfgScale, setImgCfgScale] = useState(7);
  const [imgWidth, setImgWidth] = useState(512);
  const [imgHeight, setImgHeight] = useState(512);
  const [imgSeed, setImgSeed] = useState(-1);
  const [imgSrc, setImgSrc] = useState(null);
  const [isGeneratingImg, setIsGeneratingImg] = useState(false);
  const [imgProgress, setImgProgress] = useState(null);
  const [imgMissingComponents, setImgMissingComponents] = useState(null);
  const [activeImageJobId, setActiveImageJobId] = useState(null);
  const [imgInitImage, setImgInitImage] = useState(null);
  const [imgInitImageName, setImgInitImageName] = useState(null);
  const [imgStrength, setImgStrength] = useState(0.75);
  const imgInitImageInputRef = useRef(null);

  // Video Studio State
  const [videoPrompt, setVideoPrompt] = useState('A serene waterfall in a mystical forest, cinematic motion');
  const [videoSchema, setVideoSchema] = useState(null); // { params: [...] } from GET /v1/videos/schema, null if unavailable
  const [videoParamValues, setVideoParamValues] = useState({}); // current values of the dynamic schema-driven params (all fields but prompt)
  const [videoSrc, setVideoSrc] = useState(null);
  const [isGeneratingVideo, setIsGeneratingVideo] = useState(false);
  const [videoProgress, setVideoProgress] = useState(null);
  const [modelLoadProgress, setModelLoadProgress] = useState(null); // polled from /v1/model/load-progress while isLoadingModel is true (e.g. GGUF video de-quantization)
  const [activeVideoJobId, setActiveVideoJobId] = useState(null);
  const [videoInitImage, setVideoInitImage] = useState(null);
  const [videoInitImageName, setVideoInitImageName] = useState(null);
  const videoInitImageInputRef = useRef(null);

  // 3D Model Studio State
  const [mesh3dPrompt, setMesh3dPrompt] = useState('A low-poly wooden treasure chest, game-ready asset');
  const [mesh3dImages, setMesh3dImages] = useState([]); // [{name, dataUrl}]
  const [mesh3dInputKind, setMesh3dInputKind] = useState('text');
  const [mesh3dSchema, setMesh3dSchema] = useState(null); // { <adapterId>: { params: [...] } }, null if unavailable
  const [mesh3dCfgOverrides, setMesh3dCfgOverrides] = useState({}); // saved per-model defaults, keyed by param name
  const [mesh3dParamValues, setMesh3dParamValues] = useState({}); // current values of the dynamic schema-driven params
  const [mesh3dFormat, setMesh3dFormat] = useState('glb');
  const [mesh3dTexture, setMesh3dTexture] = useState(true);
  const [mesh3dResult, setMesh3dResult] = useState(null); // {base64, format}
  const [isGeneratingMesh, setIsGeneratingMesh] = useState(false);
  const [meshProgress, setMeshProgress] = useState(null);
  const [activeMeshJobId, setActiveMeshJobId] = useState(null);
  const mesh3dImageInputRef = useRef(null);
  const mesh3dMultiImageInputRef = useRef(null);

  // Voice Studio State
  const [ttsInput, setTtsInput] = useState('Welcome to Dispos Studio.');
  const [ttsSpeed, setTtsSpeed] = useState(1.0);
  const [audioSrc, setAudioSrc] = useState(null);
  const [isGeneratingTts, setIsGeneratingTts] = useState(false);
  const [ttsProgress, setTtsProgress] = useState(null);
  const [activeTtsJobId, setActiveTtsJobId] = useState(null);

  // Transcribe (ASR) Studio State
  const [asrFileName, setAsrFileName] = useState(null);
  const [asrLanguage, setAsrLanguage] = useState('');
  const [asrText, setAsrText] = useState('');
  const [isTranscribing, setIsTranscribing] = useState(false);
  const asrFileInputRef = useRef(null);

  // Embeddings State
  const [embedModelId, setEmbedModelId] = useState('');
  const [embedInput, setEmbedInput] = useState('');
  const [embedResults, setEmbedResults] = useState(null); // { data: [{embedding, index}], model, usage }
  const [isEmbedding, setIsEmbedding] = useState(false);

  // Default values used by the fit preview and new model configuration.
  const [gpuLayers, setGpuLayers] = useState(99);
  const [contextSize, setContextSize] = useState(4096);
  const [temperature, setTemperature] = useState(0.7);
  const [topP, setTopP] = useState(0.9);
  const [systemPrompt, setSystemPrompt] = useState('');
  // Multi-model state
  const [loadedModels, setLoadedModels] = useState([]); // array of LoadedModelEntry from backend
  const [selectedModelId, setSelectedModelId] = useState(null); // which model chat uses
  const [openStudios, setOpenStudios] = useState([]); // { modelId, name, task_tags, modelPath }
  const [chatSessions, setChatSessions] = useState(() => {
    try {
      const stored = JSON.parse(localStorage.getItem('dispos.chat-sessions'));
      if (!Array.isArray(stored)) return [];
      // One-time migration: stamp studioKey on legacy sessions that predate
      // canonical keying. Older items lack modelPath; fall back to modelName.
      let migrated = false;
      const result = stored.map(item => {
        if (!item) return item;
        if (item.studioKey) return item;
        const studioKey = canonicalStudioKey(item.modelPath, item.modelName || item.modelId);
        const modelName = item.modelName
          || (item.modelPath ? item.modelPath.split(/[\\/]/).pop() : '')
          || (item.modelId ? item.modelId : '')
          || 'Local model';
        migrated = true;
        return { ...item, modelName, studioKey };
      });
      if (migrated) {
        try { localStorage.setItem('dispos.chat-sessions', JSON.stringify(result)); } catch { }
      }
      return result;
    } catch { return []; }
  });
  const [activeChatId, setActiveChatId] = useState(null);
  // The model the orchestrator's own chat requests use — set once by the
  // same auto-select logic as `selectedModelId` (see fetchLoadedModels) but
  // never reassigned by startStudio/expandToolCallToStudio/sidebar clicks,
  // which only ever touch `selectedModelId` for the currently open studio.
  const [orchestratorModelId, setOrchestratorModelId] = useState(null);
  // What the chat panel actually shows: the open studio's transcript, or the
  // orchestrator's own when no studio is open. Derived straight from
  // chatSessions — there is no separate "currently open studio" message
  // buffer, so switching studios can never write one transcript into
  // another's slot.
  const displayedMessages = activeChatId
    ? (chatSessions.find(session => session.id === activeChatId)?.messages ?? [])
    : orchestratorMessages;
  const [collapsedStudios, setCollapsedStudios] = useState(() => {
    try { return JSON.parse(localStorage.getItem('dispos.studio-collapsed')) ?? {}; } catch { return {}; }
  });
  // Whole-sidebar fold state (icon-only rail vs full labeled view).
  const [sidebarCollapsed, setSidebarCollapsed] = useState(() => {
    try { return JSON.parse(localStorage.getItem('dispos.sidebar-collapsed')) ?? false; } catch { return false; }
  });
  useEffect(() => {
    localStorage.setItem('dispos.sidebar-collapsed', JSON.stringify(sidebarCollapsed));
  }, [sidebarCollapsed]);
  const [editingSessionId, setEditingSessionId] = useState(null);
  const [editValue, setEditValue] = useState('');
  const [pendingDeleteId, setPendingDeleteId] = useState(null);
  const [pendingDeleteModelKey, setPendingDeleteModelKey] = useState(null);

  useEffect(() => {
    if (!pendingDeleteId) return;
    const id = setTimeout(() => setPendingDeleteId(null), 3500);
    return () => clearTimeout(id);
  }, [pendingDeleteId]);

  useEffect(() => {
    if (!pendingDeleteModelKey) return;
    const id = setTimeout(() => setPendingDeleteModelKey(null), 3500);
    return () => clearTimeout(id);
  }, [pendingDeleteModelKey]);

  // Rename animation: keep a displayed-name per session id, separate from
  // chat.title, so we can walk letter-by-letter from old to new on rename.
  const [displayedTitles, setDisplayedTitles] = useState({});
  const prevTitlesRef = useRef({});
  const animIntervalsRef = useRef({});
  // Session ids queued for fade-out before being dropped from chatSessions.
  const [pendingRemoveIds, setPendingRemoveIds] = useState([]);

  // Keep displayedTitles in sync with the chatSessions list (additions/removals).
  useEffect(() => {
    const sessionIds = new Set(chatSessions.map(session => session.id));
    setDisplayedTitles(current => {
      const next = { ...current };
      let changed = false;
      for (const session of chatSessions) {
        if (!(session.id in next)) {
          next[session.id] = session.title;
          changed = true;
        }
      }
      for (const id of Object.keys(next)) {
        if (!sessionIds.has(id)) {
          delete next[id];
          delete prevTitlesRef.current[id];
          if (animIntervalsRef.current[id]) {
            clearInterval(animIntervalsRef.current[id]);
            delete animIntervalsRef.current[id];
          }
          changed = true;
        }
      }
      return changed ? next : current;
    });
    for (const session of chatSessions) {
      if (!(session.id in prevTitlesRef.current)) {
        prevTitlesRef.current[session.id] = session.title;
      }
    }
  }, [chatSessions]);

  // When a session's title changes, walk from the old title to the new one
  // letter-by-letter over ~1.5s. Only fires on actual title changes — initial
  // mount and same-title re-renders are skipped via prevTitlesRef.
  useEffect(() => {
    const prevTitles = prevTitlesRef.current;
    for (const session of chatSessions) {
      const prev = prevTitles[session.id];
      if (prev === undefined || prev === session.title) continue;
      const oldTitle = prev;
      const newTitle = session.title;
      prevTitles[session.id] = session.title;

      if (animIntervalsRef.current[session.id]) {
        clearInterval(animIntervalsRef.current[session.id]);
      }

      const maxLen = Math.max(oldTitle.length, newTitle.length, 1);
      const stepMs = Math.max(40, Math.floor(1500 / maxLen));
      let step = 0;

      const tick = () => {
        step++;
        if (step > maxLen) {
          clearInterval(animIntervalsRef.current[session.id]);
          delete animIntervalsRef.current[session.id];
          setDisplayedTitles(current => ({ ...current, [session.id]: newTitle }));
          return;
        }
        const revealed = newTitle.slice(0, step);
        const remaining = oldTitle.slice(step);
        setDisplayedTitles(current => ({ ...current, [session.id]: revealed + remaining }));
      };

      animIntervalsRef.current[session.id] = setInterval(tick, stepMs);
    }
  }, [chatSessions]);

  // Clear any in-flight rename intervals when the app unmounts.
  useEffect(() => () => {
    for (const id in animIntervalsRef.current) {
      clearInterval(animIntervalsRef.current[id]);
    }
  }, []);

  const [configTarget, setConfigTarget] = useState(null);
  const [isLoadingModel, setIsLoadingModel] = useState(false);
  const [loadingModelPath, setLoadingModelPath] = useState(null); // model_path currently in-flight for /v1/model/load, drives the "loading" status dot

  // Poll for model load progress (e.g. GGUF de-quantization). The video
  // backend defers actually spawning its Python server — and thus the
  // de-quantization pass — until the first /v1/videos/generations call
  // (see video-backend's `ensure_process_started`, called from `generate()`),
  // not from /v1/model/load. So this has to poll during isGeneratingVideo
  // too, not just isLoadingModel, or the backend's real progress data (which
  // does show up on /v1/model/load-progress) never gets picked up.
  useEffect(() => {
    if (!isLoadingModel && !isGeneratingVideo) {
      setModelLoadProgress(null);
      return;
    }
    const indeterminate = { percent: -1, total: 0, step: 0, phase: 'Loading model…' };
    setModelLoadProgress(indeterminate);
    const poll = async () => {
      try {
        const res = await fetch('http://127.0.0.1:8080/v1/model/load-progress');
        const data = await res.json();
        setModelLoadProgress(data.loading
          ? { percent: data.percent, total: data.total, step: data.current, phase: data.phase }
          : (isLoadingModel ? indeterminate : null));
      } catch {
        // daemon briefly unavailable, keep showing last known progress
      }
    };
    poll();
    const interval = setInterval(poll, 400);
    return () => {
      clearInterval(interval);
      setModelLoadProgress(null);
    };
  }, [isLoadingModel, isGeneratingVideo]);
  const [installProgress, setInstallProgress] = useState(null); // active backend Python env install (auto-triggered by envNotInstalled), or null when idle; only one install runs at a time
  const [unloadingModelId, setUnloadingModelId] = useState(null);
  const [modelFitPreview, setModelFitPreview] = useState(null);
  const [detectedModels, setDetectedModels] = useState([]);
  const [modelCards, setModelCards] = useState(() => {
    const saved = localStorage.getItem('dispos-model-cards');
    return saved ? JSON.parse(saved) : [];
  });
  const [showModelPicker, setShowModelPicker] = useState(false);
  const [modelPickerSearch, setModelPickerSearch] = useState('');
  const [modelPickerTab, setModelPickerTab] = useState('local');
  const [hfSearchQuery, setHfSearchQuery] = useState('');
  const [hfSearchResults, setHfSearchResults] = useState([]);
  const [hfSearchLoading, setHfSearchLoading] = useState(false);
  const [hfSearchError, setHfSearchError] = useState(false);
  const [hfSort, setHfSort] = useState('trendingScore');
  const [hfFilters, setHfFilters] = useState([]);
  const [hfSidebarSections, setHfSidebarSections] = useState({ sort: true, modelType: true, format: true, params: true, language: false, precision: false });
  const [hfParamMinIdx, setHfParamMinIdx] = useState(0);
  const [hfParamMaxIdx, setHfParamMaxIdx] = useState(HF_PARAM_STOPS.length - 1);
  const [hfSelectedRepo, setHfSelectedRepo] = useState(null);
  const [hfRepoFiles, setHfRepoFiles] = useState([]);
  const [hfRepoFilesLoading, setHfRepoFilesLoading] = useState(false);
  const [hfRepoKind, setHfRepoKind] = useState(null);
  const [hfAutodownload, setHfAutodownload] = useState([]);
  const [hfAutodownloadReason, setHfAutodownloadReason] = useState(null);
  const [hfCollapsedFolders, setHfCollapsedFolders] = useState(() => new Set());
  const [hfFileSearch, setHfFileSearch] = useState('');
  const [hfDownloads, setHfDownloads] = useState({});
  const [componentAssignStatus, setComponentAssignStatus] = useState({});
  const [showDownloadsPanel, setShowDownloadsPanel] = useState(false);
  const [pendingCatalogJump, setPendingCatalogJump] = useState(null);
  const [hfTokenInput, setHfTokenInput] = useState('');
  const [hfTokenSaved, setHfTokenSaved] = useState(false);
  const [hfHasToken, setHfHasToken] = useState(false);

  const closeModelPicker = () => {
    setShowModelPicker(false);
    setModelPickerSearch('');
    setModelPickerTab('local');
    setHfSearchQuery('');
    setHfSearchResults([]);
    setHfSearchLoading(false);
    setHfSearchError(false);
    setHfSelectedRepo(null);
    setHfRepoFiles([]);
    setPendingCatalogJump(null);
  };

  const formatCount = (n) => {
    if (n == null) return '0';
    if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
    if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
    return `${n}`;
  };

  const formatFileSize = (bytes) => {
    if (bytes == null) return '';
    const gb = bytes / 1e9;
    if (gb >= 1) return `${gb.toFixed(2)} GB`;
    return `${(bytes / 1e6).toFixed(0)} MB`;
  };

  const formatParamCount = (n) => {
    if (n == null) return null;
    if (n >= 1e9) return `${(n / 1e9 >= 10 ? Math.round(n / 1e9) : (n / 1e9).toFixed(1))}B`;
    if (n >= 1e6) return `${(n / 1e6 >= 10 ? Math.round(n / 1e6) : (n / 1e6).toFixed(1))}M`;
    return `${n}`;
  };

  const hfParamRangeLabel = () => {
    const maxTop = HF_PARAM_STOPS.length - 1;
    const min = HF_PARAM_STOPS[hfParamMinIdx];
    const max = HF_PARAM_STOPS[hfParamMaxIdx];
    if (hfParamMinIdx === 0 && hfParamMaxIdx === maxTop) return 'Any';
    if (hfParamMinIdx === 0) return `Up to ${formatParamStop(max)}`;
    if (hfParamMaxIdx === maxTop) return `${formatParamStop(min)}+`;
    return `${formatParamStop(min)} – ${formatParamStop(max)}`;
  };

  const toggleHfFilter = (tag) => {
    setHfFilters(current => current.includes(tag) ? current.filter(t => t !== tag) : [...current, tag]);
  };

  const toggleHfSidebarSection = (key) => {
    setHfSidebarSections(current => ({ ...current, [key]: !current[key] }));
  };

  const runHfSearch = useCallback((query, sort, filters, minParams, maxParams) => {
    setHfSearchLoading(true);
    setHfSearchError(false);
    const params = new URLSearchParams({ sort });
    const trimmed = query.trim();
    if (trimmed) params.set('q', trimmed);
    if (filters.length > 0) params.set('filter', filters.join(','));
    if (minParams != null) params.set('min_params', String(minParams));
    if (maxParams != null) params.set('max_params', String(maxParams));
    fetch(`http://127.0.0.1:8080/v1/model/hf-search?${params.toString()}`)
      .then(res => res.ok ? res.json() : Promise.reject(new Error('search failed')))
      .then(results => setHfSearchResults(Array.isArray(results) ? results : []))
      .catch(() => {
        setHfSearchResults([]);
        setHfSearchError(true);
      })
      .finally(() => setHfSearchLoading(false));
  }, []);

  const hfParamBounds = () => {
    const maxTop = HF_PARAM_STOPS.length - 1;
    const minParams = hfParamMinIdx > 0 ? Math.round(HF_PARAM_STOPS[hfParamMinIdx] * 1e9) : null;
    const maxParams = hfParamMaxIdx < maxTop ? Math.round(HF_PARAM_STOPS[hfParamMaxIdx] * 1e9) : null;
    return [minParams, maxParams];
  };

  useEffect(() => {
    if (modelPickerTab !== 'discover') return;
    const [minParams, maxParams] = hfParamBounds();
    const handle = setTimeout(() => runHfSearch(hfSearchQuery, hfSort, hfFilters, minParams, maxParams), 500);
    return () => clearTimeout(handle);
  }, [hfSearchQuery, hfSort, hfFilters, hfParamMinIdx, hfParamMaxIdx, modelPickerTab, runHfSearch]);

  useEffect(() => {
    fetch('http://127.0.0.1:8080/v1/model/hf-token')
      .then(res => res.json())
      .then(data => setHfHasToken(data.has_token))
      .catch(() => { });
  }, []);

  const saveHfToken = () => {
    fetch('http://127.0.0.1:8080/v1/model/hf-token', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token: hfTokenInput }),
    })
      .then(res => res.json())
      .then(data => {
        setHfHasToken(data.has_token);
        setHfTokenSaved(true);
        setHfTokenInput('');
        setTimeout(() => setHfTokenSaved(false), 3000);
      })
      .catch(() => { });
  };

  const selectHfRepo = (repoId) => {
    if (hfSelectedRepo === repoId) {
      setHfSelectedRepo(null);
      setHfRepoFiles([]);
      setHfRepoKind(null);
      setHfAutodownload([]);
      setHfAutodownloadReason(null);
      return;
    }
    setHfSelectedRepo(repoId);
    setHfRepoFiles([]);
    setHfRepoKind(null);
    setHfAutodownload([]);
    setHfAutodownloadReason(null);
    setHfCollapsedFolders(new Set());
    setHfFileSearch('');
    setHfDownloads(current => {
      const next = { ...current };
      for (const key of Object.keys(next)) {
        if (next[key].status === 'complete') delete next[key];
      }
      return next;
    });
    setHfRepoFilesLoading(true);
    fetch(`http://127.0.0.1:8080/v1/model/hf-files?repo=${encodeURIComponent(repoId)}`)
      .then(res => res.ok ? res.json() : null)
      .then(data => {
        const files = Array.isArray(data) ? data : (Array.isArray(data?.files) ? data.files : []);
        setHfRepoFiles(files);
        setHfRepoKind(Array.isArray(data) ? null : (data?.kind ?? null));
        setHfAutodownload(Array.isArray(data?.autodownload) ? data.autodownload : []);
        setHfAutodownloadReason(Array.isArray(data) ? null : (data?.autodownload_reason ?? null));
      })
      .catch(() => setHfRepoFiles([]))
      .finally(() => setHfRepoFilesLoading(false));
  };

  const startHfDownload = (repoId, filename, targetDir, targetFilename) => {
    const key = `${repoId}::${filename}`;
    setHfDownloads(current => ({ ...current, [key]: { status: 'downloading', downloaded_bytes: 0, total_bytes: 0, repo: repoId, filename } }));
    fetch('http://127.0.0.1:8080/v1/model/hf-download', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        repo: repoId,
        filename,
        ...(targetDir ? { target_dir: targetDir } : {}),
        ...(targetFilename ? { target_filename: targetFilename } : {}),
      }),
    }).catch(() => {
      setHfDownloads(current => ({ ...current, [key]: { ...current[key], status: 'error' } }));
    });
  };

  // Used by video pipeline components, whose expected filenames are fixed
  // and known ahead of time (LTX's tokenizer/scheduler/text-encoder file
  // lists) — so physically placing the picked file at the exact expected
  // path/name is unambiguous and survives restarts via the normal
  // exact-filename detection.
  const assignComponentFile = async (comp) => {
    const targetPath = comp.target_path;
    const defaultPath = targetPath ? window.require('path').dirname(targetPath) : undefined;
    const sourcePath = await browseForFile(defaultPath, undefined, ['openFile']);
    if (!sourcePath) return;
    const path = window.require('path');
    const destination = comp.resolved_path
      ? targetPath
      : path.join(targetPath, comp.source?.target_filename || path.basename(sourcePath));
    setComponentAssignStatus(current => ({ ...current, [targetPath]: { status: 'assigning' } }));
    try {
      const res = await fetch('http://127.0.0.1:8080/v1/model/assign-component', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source_path: sourcePath, target_path: destination }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok || data.error) {
        setComponentAssignStatus(current => ({ ...current, [targetPath]: { status: 'error', error: data.error || 'Assign failed' } }));
        return;
      }
      setComponentAssignStatus(current => {
        const next = { ...current };
        delete next[targetPath];
        return next;
      });
      fetchCatalog();
    } catch (err) {
      setComponentAssignStatus(current => ({ ...current, [targetPath]: { status: 'error', error: String(err) } }));
    }
  };

  // Used by image (sd-backend) sibling components. These are matched by a
  // fuzzy filename hint (e.g. "qwen3", "flux2-vae"), so a user-picked file
  // usually won't match the pattern and copying it into place wouldn't be
  // found again on restart — this records the exact chosen path as an
  // override instead, which the backend checks before falling back to the
  // fuzzy search.
  const assignImageComponentOverride = async (comp) => {
    const key = comp.target_path;
    const defaultPath = key ? window.require('path').dirname(key) : undefined;
    const sourcePath = await browseForFile(defaultPath, undefined, ['openFile']);
    if (!sourcePath) return;
    setComponentAssignStatus(current => ({ ...current, [key]: { status: 'assigning' } }));
    try {
      const res = await fetch('http://127.0.0.1:8080/v1/model/component-override', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_path: configTarget?.model_path, kind_name: comp.kind_name, source_path: sourcePath }),
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok || data.error) {
        setComponentAssignStatus(current => ({ ...current, [key]: { status: 'error', error: data.error || 'Assign failed' } }));
        return;
      }
      setComponentAssignStatus(current => {
        const next = { ...current };
        delete next[key];
        return next;
      });
      fetchCatalog();
    } catch (err) {
      setComponentAssignStatus(current => ({ ...current, [key]: { status: 'error', error: String(err) } }));
    }
  };

  const startHfDownloadAll = (repoId) => {
    hfRepoFiles.forEach(file => {
      if (hfDownloads[`${repoId}::${file.filename}`]?.status === 'complete') return;
      startHfDownload(repoId, file.filename);
    });
  };

  const startHfAutodownload = (repoId) => {
    hfAutodownload.forEach(filename => {
      if (hfDownloads[`${repoId}::${filename}`]?.status === 'complete') return;
      startHfDownload(repoId, filename);
    });
  };

  const cancelHfDownload = (repo, filename) => {
    fetch('http://127.0.0.1:8080/v1/model/hf-download/cancel', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ repo, filename }),
    }).catch(() => { });
  };

  // "Find in catalog" from the downloads tracker: open the catalog on Discover,
  // seed the search with the repo, then let the effects below drive selection
  // and scroll/highlight once the async search + file-list fetch settle.
  const jumpToCatalogEntry = (repo, filename) => {
    setShowDownloadsPanel(false);
    setShowModelPicker(true);
    setModelPickerTab('discover');
    setHfSearchQuery(repo);
    setPendingCatalogJump({ repo, filename });
  };

  // Once the Discover search settles, auto-select the matching repo card.
  // Fails soft: if the repo isn't in the (capped) search results, drop the
  // pending jump and leave the catalog open with the query seeded.
  useEffect(() => {
    if (!pendingCatalogJump || modelPickerTab !== 'discover' || hfSearchLoading) return;
    if (hfSelectedRepo === pendingCatalogJump.repo) return;
    const match = hfSearchResults.find(r => (r.id || `${r.author}/${r.modelId}`) === pendingCatalogJump.repo);
    if (match) {
      selectHfRepo(pendingCatalogJump.repo);
    } else {
      setPendingCatalogJump(null);
    }
  }, [pendingCatalogJump, modelPickerTab, hfSearchLoading, hfSearchResults, hfSelectedRepo]);

  // Once the selected repo's file list settles, scroll to and briefly
  // highlight the target file row. Fails soft if the file isn't found.
  useEffect(() => {
    if (!pendingCatalogJump || hfSelectedRepo !== pendingCatalogJump.repo || hfRepoFilesLoading) return;
    const { filename } = pendingCatalogJump;
    const raf = requestAnimationFrame(() => {
      const escaped = window.CSS?.escape ? window.CSS.escape(filename) : filename;
      const el = document.querySelector(`[data-filename="${escaped}"]`);
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
        el.classList.add('hf-file-row-highlight');
        setTimeout(() => el.classList.remove('hf-file-row-highlight'), 2000);
      }
      setPendingCatalogJump(null);
    });
    return () => cancelAnimationFrame(raf);
  }, [pendingCatalogJump, hfSelectedRepo, hfRepoFilesLoading, hfRepoFiles]);

  const fetchCatalog = useCallback(() => {
    fetch('http://127.0.0.1:8080/v1/model/catalog')
      .then(res => res.ok ? res.json() : [])
      .then(models => setDetectedModels(Array.isArray(models) ? models : []))
      .catch(() => setDetectedModels([]));
  }, []);

  // Seed hfDownloads from the daemon on mount so downloads started in a
  // previous popup session (or before this page loaded) are recovered.
  useEffect(() => {
    fetch('http://127.0.0.1:8080/v1/model/hf-download/status')
      .then(res => res.ok ? res.json() : {})
      .then(status => setHfDownloads(current => ({ ...status, ...current })))
      .catch(() => { });
  }, []);

  // Global download-status polling: runs independent of the catalog modal
  // being open so the tracker stays accurate even after it's closed. Only
  // polls while something is actively downloading.
  useEffect(() => {
    const activeDownloads = Object.entries(hfDownloads).filter(([, info]) => info.status === 'downloading');
    if (activeDownloads.length === 0) return;
    const interval = setInterval(() => {
      fetch('http://127.0.0.1:8080/v1/model/hf-download/status')
        .then(res => res.ok ? res.json() : {})
        .then(status => {
          const anyCompleted = Object.entries(status).some(
            ([key, info]) =>
              info.status === 'complete' &&
              hfDownloads[key] && hfDownloads[key].status !== 'complete'
          );
          setHfDownloads(current => {
            const next = { ...current };
            for (const [key, info] of Object.entries(status)) {
              next[key] = { ...next[key], ...info };
            }
            return next;
          });
          // A freshly-downloaded model won't be in the local catalog yet; refetch
          // so it shows up under "my models" without a manual refresh.
          if (anyCompleted) fetchCatalog();
        })
        .catch(() => { });
    }, 2000);
    return () => clearInterval(interval);
  }, [hfDownloads, fetchCatalog]);

  // Auto-retry image generation once every downloadable missing component
  // finishes downloading. Clearing imgMissingComponents here (rather than
  // inside handleGenerateImage before this effect re-runs) is what prevents
  // an infinite retry loop if the retry fails again.
  useEffect(() => {
    if (!imgMissingComponents || imgMissingComponents.length === 0) return;
    const downloadable = imgMissingComponents.filter(c => c.source);
    if (downloadable.length === 0) return;
    const allDone = downloadable.every(c => hfDownloads[`${c.source.repo}::${c.source.filename}`]?.status === 'complete');
    if (allDone) {
      setImgMissingComponents(null);
      handleGenerateImage();
    }
  }, [hfDownloads, imgMissingComponents]);

  useEffect(() => {
    localStorage.setItem('dispos-model-cards', JSON.stringify(modelCards));
  }, [modelCards]);

  useEffect(() => {
    fetch('http://127.0.0.1:8080/v1/studio/models', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ paths: modelCards.map(c => c.modelPath) }),
    }).catch(() => { });
  }, [modelCards]);

  useEffect(() => {
    localStorage.setItem('dispos.chat-sessions', JSON.stringify(chatSessions));
  }, [chatSessions]);

  useEffect(() => {
    localStorage.setItem('dispos.studio-collapsed', JSON.stringify(collapsedStudios));
  }, [collapsedStudios]);

  // Model preset catalog
  const MODEL_PRESETS = [
    {
      name: 'Qwen 3.5 0.8B',
      tag: 'Q8_0 · 0.8B',
      path: 'models\\Qwen3.5-0.8B-Q8_0.gguf',
      defaultGpu: 99, defaultCtx: 4096, defaultTemp: 0.7, defaultTopP: 0.9,
    },
    {
      name: 'DeepSeek R1 1.5B',
      tag: 'Q4_K_M · 1.5B',
      path: 'C:\\Users\\adem2\\.lmstudio\\models\\deepseek-ai\\DeepSeek-R1-Distill-Qwen-1.5B-GGUF\\DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf',
      defaultGpu: 99, defaultCtx: 4096, defaultTemp: 0.6, defaultTopP: 0.95,
    },
  ];

  const openConfigPanel = (card) => {
    const s = card.settings || {};
    const maxContextSize = card.max_context_size ?? null;
    setConfigTarget({
      cardId: card.id,
      model_path: card.modelPath,
      name: card.name,
      task_tags: card.task_tags || [],
      mesh_input_kinds: card.mesh_input_kinds || detectedModels.find(m => m.path === card.modelPath)?.mesh_input_kinds || null,
      gpu_layers: s.gpu_layers ?? 99,
      context_size: maxContextSize ? Math.min(s.context_size ?? 4096, maxContextSize) : (s.context_size ?? 4096),
      max_context_size: maxContextSize,
      temperature: s.temperature ?? 0.7,
      top_p: s.top_p ?? 0.9,
      system_prompt: s.system_prompt ?? '',
      steps: s.steps,
      cfg_scale: s.cfg_scale,
      width: s.width,
      height: s.height,
      seed: s.seed,
      guidance_scale: s.guidance_scale,
      texture: s.texture,
      speed: s.speed,
      mmproj_path: s.mmproj_path || card.mmproj_path || detectedModels.find(m => m.path === card.modelPath)?.mmproj_path || '',
      mtp_path: s.mtp_path || card.mtp_path || detectedModels.find(m => m.path === card.modelPath)?.mtp_path || '',
      mtp_enabled: s.mtp_enabled,
      spec_draft_n_max: s.spec_draft_n_max,
      spec_draft_p_min: s.spec_draft_p_min,
      text_encoder_override_path: s.text_encoder_override_path || card.text_encoder_override_path || '',
      vae_override_path: s.vae_override_path || card.vae_override_path || '',
      mesh_vae_path: s.mesh_vae_path || card.mesh_vae_path || detectedModels.find(m => m.path === card.modelPath)?.mesh_vae_path || '',
      mesh_texgen_path: s.mesh_texgen_path || card.mesh_texgen_path || detectedModels.find(m => m.path === card.modelPath)?.mesh_texgen_path || '',
      // Generic sibling-component list (covers Hunyuan3D subfolders and flat
      // multi-file repos like TRELLIS). Falls back to the legacy VAE/texgen
      // fields above for older cached cards that predate this field.
      mesh_components: s.mesh_components || card.mesh_components || detectedModels.find(m => m.path === card.modelPath)?.mesh_components || [
        ...(s.mesh_vae_path || card.mesh_vae_path ? [{ label: 'VAE', path: s.mesh_vae_path || card.mesh_vae_path }] : []),
        ...(s.mesh_texgen_path || card.mesh_texgen_path ? [{ label: 'Texture/Paint', path: s.mesh_texgen_path || card.mesh_texgen_path }] : []),
      ],
      image_components: detectedModels.find(m => m.path === card.modelPath)?.image_components || null,
      video_components: detectedModels.find(m => m.path === card.modelPath)?.video_components || null,
      vae_path: s.vae_path || card.vae_path || detectedModels.find(m => m.path === card.modelPath)?.vae_path || '',
    });
  };

  const buildSettingsFromConfigTarget = (cfg) => ({
    gpu_layers: cfg?.gpu_layers,
    context_size: cfg?.context_size,
    mmproj_path: cfg?.mmproj_path || undefined,
    mtp_path: cfg?.mtp_path || undefined,
    mtp_enabled: cfg?.mtp_enabled,
    spec_draft_n_max: cfg?.spec_draft_n_max,
    spec_draft_p_min: cfg?.spec_draft_p_min,
    text_encoder_override_path: cfg?.text_encoder_override_path || undefined,
    vae_override_path: cfg?.vae_override_path || undefined,
    temperature: cfg?.temperature,
    top_p: cfg?.top_p,
    system_prompt: cfg?.system_prompt || '',
    steps: cfg?.steps,
    cfg_scale: cfg?.cfg_scale,
    width: cfg?.width,
    height: cfg?.height,
    seed: cfg?.seed,
    guidance_scale: cfg?.guidance_scale,
    texture: cfg?.texture,
    speed: cfg?.speed,
  });

  const closeConfigPanel = () => {
    if (configTarget?.cardId) {
      const settings = buildSettingsFromConfigTarget(configTarget);
      setModelCards(current => current.map(item => item.id === configTarget.cardId ? { ...item, settings } : item));
    }
    setConfigTarget(null);
  };

  const fetchLoadedModels = useCallback(async () => {
    try {
      const res = await fetch('http://127.0.0.1:8080/v1/model/list');
      const data = await res.json();
      if (Array.isArray(data)) {
        setLoadedModels(data);
        // Auto-select first model if none selected
        if (data.length > 0 && !selectedModelId) {
          setSelectedModelId(data[0].model_id);
        }
        if (data.length === 0) setSelectedModelId(null);
        // Same auto-select, kept independent of selectedModelId so opening a
        // studio (which claims selectedModelId for that model) never leaves
        // the orchestrator's own chat requests pointed at the wrong model.
        // Scoped to category "chat" — previously this grabbed data[0]
        // unconditionally, so if a non-chat model (image/video/etc.) happened
        // to be first in the daemon's list, the orchestrator (and the Chat
        // Studio status strip's no-session fallback, which reads
        // orchestratorModelId) would silently point at the wrong modality's
        // model instead of showing "No Model Loaded".
        // Re-point whenever the currently-tracked model has fallen out of
        // loadedModels (unloaded, or replaced by a reload that issued a new
        // timestamped model_id) — not just when orchestratorModelId was never
        // set. Previously this only assigned on the very first successful
        // fetch (`!orchestratorModelId`), so once set it never updated again:
        // if a second chat model was loaded, then the *tracked* one got
        // unloaded (a common flow once the orchestrator itself starts
        // swapping models via run_model), orchestratorModelId kept pointing
        // at the now-gone model_id forever — data.find() always finding some
        // *other* chat model doesn't help since the id it looks up by no
        // longer exists in loadedModels, so activeChatLoaded/activeChatStudio
        // permanently read null and the orchestrator's own status strip blanked
        // out until every last chat model happened to be unloaded at once.
        const firstChatModel = data.find(model => categoryForTags(model.task_tags) === 'chat');
        const trackedStillLoaded = orchestratorModelId && data.some(model => model.model_id === orchestratorModelId);
        if (firstChatModel && !trackedStillLoaded) {
          setOrchestratorModelId(firstChatModel.model_id);
        } else if (!firstChatModel) {
          setOrchestratorModelId(null);
        }
      }
    } catch { }
  }, [selectedModelId, orchestratorModelId]);

  const handleLoadModel = async (configuration = configTarget) => {
    if (!configuration?.model_path?.trim() || isLoadingModel) return;
    setIsLoadingModel(true);
    setLoadingModelPath(configuration.model_path);
    try {
      const res = await fetch('http://127.0.0.1:8080/v1/model/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_path: configuration.model_path,
          gpu_layers: configuration.gpu_layers,
          context_size: configuration.context_size,
          mmproj_path: configuration.mmproj_path || undefined,
          mtp_path: configuration.mtp_path || undefined,
          mtp_enabled: configuration.mtp_enabled,
          spec_draft_n_max: configuration.spec_draft_n_max,
          spec_draft_p_min: configuration.spec_draft_p_min,
          text_encoder_override_path: configuration.text_encoder_override_path || undefined,
          vae_override_path: configuration.vae_override_path || undefined,
        }),
      });
      const text = await res.text();
      let data = null;
      try { data = text ? JSON.parse(text) : null; } catch { /* plain-text error body */ }
      if (res.ok && data?.model_id) {
        await fetchLoadedModels();
        fetchSystemInfo();
        setTimeout(fetchSystemInfo, 1500);
        return data.model_id;
      } else {
        pushError('Failed to load model: ' + (data?.error || text || res.statusText));
        return null;
      }
    } catch (err) {
      pushError('Error loading model: ' + err.message);
      return null;
    } finally {
      setIsLoadingModel(false);
      setLoadingModelPath(null);
    }
  };

  // Auto-triggered when a generation call fails with envNotInstalled (the
  // backend's lazily-spawned Python subprocess is missing deps). Kicks off
  // the backend's Python env install, polls its status every 200ms (same
  // convention as runGenerationJob) while it's running, and stops polling
  // once it lands on complete/error. On complete, calls `onInstalled` (if
  // given) to retry the original operation exactly once; on error, surfaces
  // the failure via pushError instead of retrying.
  const installEnvAndRetry = async (backend, onInstalled) => {
    try {
      const res = await fetch(`http://127.0.0.1:8080/v1/env/install/${backend}`, { method: 'POST' });
      setInstallProgress(res.ok ? await res.json() : { backend, status: 'running', phase: 'Starting install...', step: 0, total: 0, percent: -1 });
    } catch (err) {
      pushError('Failed to start environment install: ' + err.message);
      return null;
    }
    return new Promise((resolve) => {
      const interval = setInterval(() => {
        fetch(`http://127.0.0.1:8080/v1/env/install/${backend}/status`)
          .then(res => (res.ok ? res.json() : null))
          .then(async record => {
            if (!record) return;
            if (record.status !== 'complete' && record.status !== 'error') {
              setInstallProgress(record);
              return;
            }
            clearInterval(interval);
            setInstallProgress(null);
            if (record.status === 'complete') {
              resolve(onInstalled ? await onInstalled() : null);
            } else {
              pushError('Environment install failed: ' + (record.message || 'unknown error'));
              resolve(null);
            }
          })
          .catch(() => { });
      }, 200);
    });
  };

  // Seed the relevant studio's live controls from the model's configured
  // defaults, so settings chosen in the Configure sidebar actually drive
  // generation. Fallbacks match each studio's own defaults.
  const applyModelDefaults = (cfg) => {
    if (!cfg) return;
    switch (categoryForTags(cfg.task_tags)) {
      case 'chat':
        setTemperature(cfg.temperature ?? 0.7);
        setTopP(cfg.top_p ?? 0.9);
        setSystemPrompt(cfg.system_prompt ?? '');
        break;
      case 'image':
        setImgSteps(cfg.steps ?? 25);
        setImgCfgScale(cfg.cfg_scale ?? 7);
        setImgWidth(cfg.width ?? 512);
        setImgHeight(cfg.height ?? 512);
        setImgSeed(cfg.seed ?? -1);
        break;
      case 'tts':
        setTtsSpeed(cfg.speed ?? 1.0);
        break;
      case 'mesh3d':
        // These are the only param names the "Configure Model" sidebar saves;
        // they're applied as overrides on top of the schema's own defaults
        // only for adapters whose dynamic params actually use these names.
        setMesh3dCfgOverrides({ steps: cfg.steps ?? 64, guidance_scale: cfg.guidance_scale ?? 15, seed: cfg.seed ?? -1 });
        setMesh3dTexture(cfg.texture ?? true);
        break;
      default:
        break;
    }
  };

  // Single load path shared by the card "Load" button and the Configure
  // sidebar. `overrideSettings` (from the sidebar) is persisted onto the card
  // so a later card-load uses the same values. Load-time options come from the
  // settings; generation defaults are pushed into the studios via
  // applyModelDefaults.
  const loadCardModel = async (card, overrideSettings) => {
    const s = overrideSettings ?? card.settings ?? {};
    const contextSize = card.max_context_size
      ? Math.min(s.context_size ?? 4096, card.max_context_size)
      : (s.context_size ?? 4096);
    const mmproj = s.mmproj_path || card.mmproj_path || detectedModels.find(m => m.path === card.modelPath)?.mmproj_path || undefined;
    const mtp = s.mtp_path || card.mtp_path || detectedModels.find(m => m.path === card.modelPath)?.mtp_path || undefined;
    const textEncoderOverride = s.text_encoder_override_path || card.text_encoder_override_path || undefined;
    const vaeOverride = s.vae_override_path || card.vae_override_path || undefined;
    const modelId = await handleLoadModel({
      model_path: card.modelPath,
      gpu_layers: s.gpu_layers ?? 99,
      context_size: contextSize,
      mmproj_path: mmproj,
      mtp_path: mtp,
      mtp_enabled: s.mtp_enabled,
      spec_draft_n_max: s.spec_draft_n_max,
      spec_draft_p_min: s.spec_draft_p_min,
      text_encoder_override_path: textEncoderOverride,
      vae_override_path: vaeOverride,
    });
    if (modelId) {
      applyModelDefaults({ ...s, task_tags: card.task_tags });
      setModelCards(current => current.map(item => item.id === card.id ? { ...item, settings: s, loadedModelId: modelId } : item));
    }
    return modelId;
  };

  const handleUnloadModel = async (modelId) => {
    if (!modelId) return;
    setUnloadingModelId(modelId);
    try {
      const res = await fetch('http://127.0.0.1:8080/v1/model/unload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_id: modelId }),
      });
      if (res.ok) {
        await fetchLoadedModels();
        // Keep the openStudios entry after unload (don't drop it) — the
        // status strip needs to keep showing this studio's associated model
        // name with an "idle" dot rather than reverting to "no model".
        if (selectedModelId === modelId) {
          setSelectedModelId(null);
        }
        fetchSystemInfo();
        setTimeout(fetchSystemInfo, 1500);
      }
    } catch (err) {
      pushError('Error ejecting model: ' + err.message);
    } finally {
      setUnloadingModelId(null);
    }
  };

  const handleOpenModelFolder = async (modelPath) => {
    try {
      const res = await fetch('http://127.0.0.1:8080/v1/model/open-folder', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: modelPath }),
      });
      if (!res.ok) {
        const text = await res.text();
        let data = null;
        try { data = text ? JSON.parse(text) : null; } catch { /* plain-text error body */ }
        pushError('Failed to open folder: ' + (data?.error || text || res.statusText));
      }
    } catch (err) {
      pushError('Error opening folder: ' + err.message);
    }
  };

  const handleDeleteCatalogModel = async (model) => {
    if (!window.confirm(`Delete "${model.name}" permanently?`)) return;
    try {
      const res = await fetch('http://127.0.0.1:8080/v1/model/delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: model.path, mmproj_path: model.mmproj_path || undefined, mtp_path: model.mtp_path || undefined }),
      });
      if (res.ok) {
        fetchCatalog();
        setModelCards(current => current.filter(card => card.modelPath !== model.path));
      } else {
        const text = await res.text();
        let data = null;
        try { data = text ? JSON.parse(text) : null; } catch { /* plain-text error body */ }
        pushError('Failed to delete model: ' + (data?.error || text || res.statusText));
      }
    } catch (err) {
      pushError('Error deleting model: ' + err.message);
    }
  };

  useEffect(() => {
    fetchLoadedModels();
    const interval = setInterval(fetchLoadedModels, appSettings.refreshSeconds * 1000);
    return () => clearInterval(interval);
  }, [appSettings.refreshSeconds, fetchLoadedModels]);

  useEffect(() => {
    fetchCatalog();
  }, [fetchCatalog]);

  const selectedModel = loadedModels.find(model => model.model_id === selectedModelId) ?? null;
  const selectedCatalogModel = detectedModels.find(model => model.path === selectedModel?.model_path);
  const acceptsImageInput = selectedModel?.image_input_available === true || selectedCatalogModel?.image_input_available === true;
  const cfgCategory = categoryForTags(configTarget?.task_tags);
  const openModelStudio = (model) => {
    const category = categoryForTags(model.task_tags);
    const studio = category === 'tts' ? 'audio' : category === 'audio' ? 'transcribe' : category;
    setActiveTab(['chat', 'image', 'audio', 'transcribe', 'video', 'mesh3d', 'embeddings'].includes(studio) ? studio : 'chat');
  };
  // Resolves the 4-state status-strip dot for a studio: 'online' (loaded &
  // running), 'loading' (a /v1/model/load request is in-flight for this exact
  // model path), 'idle' (selected for this studio but not currently loaded),
  // or 'missing' (the selected model no longer exists in the catalog).
  const modelDotState = (studio, loaded) => {
    if (loaded) return 'online';
    if (!studio) return null;
    if (studio.modelPath && isLoadingModel && loadingModelPath === studio.modelPath) return 'loading';
    return detectedModels.some(model => model.path === studio.modelPath) ? 'idle' : 'missing';
  };

  // The Chat Studio status strip must track whatever transcript is actually
  // on screen — the active chat session if one is selected, or the
  // orchestrator's own model when no session is (activeChatId === null),
  // exactly mirroring the `displayedMessages` logic above. Previously this
  // was derived from `openStudios` (the most recently *opened* studio),
  // which went stale/blank the moment the user switched to a different
  // session or back to the orchestrator via the sidebar — openChatSession
  // only ever calls setActiveChatId/openModelStudio, it never touches
  // openStudios, so the strip kept showing whatever studio was opened last
  // instead of the session currently being viewed.
  const activeChatSession = activeChatId ? (chatSessions.find(session => session.id === activeChatId) ?? null) : null;

  // Which model a studio "belongs to", for its status strip. The sidebar
  // STUDIOS list is built from `chatSessions`, which is persisted to
  // localStorage and holds an entry for every modality (startStudio creates
  // one whatever the category) — so that, not `openStudios` (a plain
  // useState that starts empty again on every launch), is what survives an
  // app restart. Resolving only against openStudios/loadedModels meant a
  // studio whose model isn't currently loaded read "No Model Loaded" until
  // the user loaded it by hand, even though the sidebar was still listing
  // it. Priority: the session actually on screen, then a studio opened this
  // run, then anything of that category currently running, then the most
  // recently used remembered session — so a cold start still names its model
  // and shows an idle dot instead of blanking out.
  const studioForCategory = (category) => {
    const matches = (tags) => categoryForTags(tags) === category;
    const source = (activeChatSession && matches(activeChatSession.task_tags) ? activeChatSession : null)
      ?? [...openStudios].reverse().find(studio => matches(studio.task_tags))
      ?? loadedModels.find(model => matches(model.task_tags))
      ?? [...chatSessions].sort((a, b) => chatLastModified(b) - chatLastModified(a)).find(session => matches(session.task_tags));
    if (!source) return undefined;
    return {
      modelId: source.modelId ?? source.model_id,
      name: source.modelName ?? source.name ?? source.model_path?.split('\\').pop() ?? 'Local model',
      task_tags: source.task_tags,
      modelPath: source.modelPath ?? source.model_path,
    };
  };

  const activeChatLoaded = activeChatSession
    ? (loadedModels.find(model => model.model_id === activeChatSession.modelId) ?? loadedModels.find(model => model.model_path === activeChatSession.modelPath))
    : (orchestratorModelId ? loadedModels.find(model => model.model_id === orchestratorModelId) : null);
  const activeChatStudio = activeChatSession
    ? { modelId: activeChatSession.modelId, name: activeChatSession.modelName, task_tags: activeChatSession.task_tags, modelPath: activeChatSession.modelPath }
    : (activeChatLoaded
      ? { modelId: activeChatLoaded.model_id, name: activeChatLoaded.model_path?.split('\\').pop(), task_tags: activeChatLoaded.task_tags, modelPath: activeChatLoaded.model_path }
      : studioForCategory('chat'));

  // Most recently opened loaded model with category "mesh3d" drives which
  // input kinds the 3D Model Studio panel adapts to.
  const activeMesh3dStudio = studioForCategory('mesh3d');
  const activeMesh3dLoaded = activeMesh3dStudio
    ? (loadedModels.find(model => model.model_id === activeMesh3dStudio.modelId) ?? loadedModels.find(model => model.model_path === activeMesh3dStudio.modelPath))
    : null;
  const activeMesh3dCatalog = activeMesh3dLoaded ? detectedModels.find(model => model.path === activeMesh3dLoaded.model_path) : null;
  const mesh3dAvailableKinds = activeMesh3dLoaded?.mesh_input_kinds?.length
    ? activeMesh3dLoaded.mesh_input_kinds
    : (activeMesh3dCatalog?.mesh_input_kinds?.length ? activeMesh3dCatalog.mesh_input_kinds : ['text', 'image', 'multi_image']);
  useEffect(() => {
    if (!mesh3dAvailableKinds.includes(mesh3dInputKind)) setMesh3dInputKind(mesh3dAvailableKinds[0]);
  }, [mesh3dAvailableKinds.join(','), mesh3dInputKind]);

  // Per-adapter param schema, fetched so the mesh3d panel can render its
  // tunable params generically instead of hardcoding fields per model.
  // Re-fetched whenever the active mesh3d model changes (not just once on
  // app mount) because the daemon's schema endpoint 503s until a mesh3d
  // model's threed_server.py subprocess is actually running — mounting is
  // almost always before any 3D model has been loaded. Polls with retry
  // (mirrors the video schema effect below) so the params render as soon as
  // the subprocess comes up, instead of requiring a generation attempt to
  // re-trigger this effect.
  useEffect(() => {
    if (!activeMesh3dLoaded?.model_path && !activeMesh3dCatalog?.path) {
      setMesh3dSchema(null);
      return;
    }
    let cancelled = false;
    const poll = async () => {
      try {
        const res = await fetch('http://127.0.0.1:8080/v1/models3d/schema');
        if (!res.ok) throw new Error('schema fetch failed');
        const schema = await res.json();
        if (!cancelled) setMesh3dSchema(schema);
        return;
      } catch {
        if (!cancelled) setTimeout(poll, 1500);
      }
    };
    setMesh3dSchema(null);
    poll();
    return () => { cancelled = true; };
  }, [activeMesh3dLoaded?.model_path, activeMesh3dCatalog?.path]);
  const mesh3dAdapterId = detectMesh3dAdapterId(activeMesh3dLoaded?.model_path || activeMesh3dCatalog?.path);
  const mesh3dAdapterParams = mesh3dSchema?.[mesh3dAdapterId]?.params || [];
  // Re-seed the dynamic param values whenever the active adapter (or the
  // schema itself) changes, applying any saved per-model overrides first.
  useEffect(() => {
    const defaults = {};
    mesh3dAdapterParams.forEach(p => {
      defaults[p.name] = mesh3dCfgOverrides[p.name] ?? meshParamDefaultValue(p);
    });
    setMesh3dParamValues(defaults);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mesh3dAdapterId, mesh3dSchema, mesh3dCfgOverrides]);

  // Most recently opened loaded model with category "image" drives whether
  // the Image Studio panel exposes the optional reference-image (img2img) UI.
  const activeImageStudio = studioForCategory('image');
  const activeImageLoaded = activeImageStudio
    ? (loadedModels.find(model => model.model_id === activeImageStudio.modelId) ?? loadedModels.find(model => model.model_path === activeImageStudio.modelPath))
    : null;
  const imageStudioSupportsImg2Img = !!activeImageStudio?.task_tags?.includes('image-to-image');

  // Most recently opened loaded model with category "video" drives the
  // Video Studio panel's dynamic param schema.
  const activeVideoStudio = studioForCategory('video');
  const activeVideoLoaded = activeVideoStudio
    ? (loadedModels.find(model => model.model_id === activeVideoStudio.modelId) ?? loadedModels.find(model => model.model_path === activeVideoStudio.modelPath))
    : null;

  // Most recently opened loaded model with category "tts"/"audio" (ASR) —
  // same openStudios + categoryForTags mechanism as image/video/mesh3d above.
  const activeTtsStudio = studioForCategory('tts');
  const activeTtsLoaded = activeTtsStudio
    ? (loadedModels.find(model => model.model_id === activeTtsStudio.modelId) ?? loadedModels.find(model => model.model_path === activeTtsStudio.modelPath))
    : null;
  const activeAsrStudio = studioForCategory('audio');
  const activeAsrLoaded = activeAsrStudio
    ? (loadedModels.find(model => model.model_id === activeAsrStudio.modelId) ?? loadedModels.find(model => model.model_path === activeAsrStudio.modelPath))
    : null;

  // Embeddings has no "opened studio" of its own — it uses the explicit
  // model picker below, defaulting to the first loaded chat model. Both
  // branches are already scoped to category "chat" so this never leaks a
  // model from another modality.
  const activeEmbedLoaded = embedModelId
    ? loadedModels.find(model => model.model_id === embedModelId)
    : loadedModels.find(model => categoryForTags(model.task_tags) === 'chat');
  // embedModelId can go stale (its model was unloaded elsewhere) without a
  // matching loadedModels entry to recover a path/name from — fall back to
  // showing the raw id so the strip still names a model instead of going
  // blank, with an "idle" dot since we can't verify catalog presence by id.
  const activeEmbedDotState = activeEmbedLoaded ? 'online' : (embedModelId ? 'idle' : null);

  // Param schema reflecting what the loaded video pipeline's `__call__`
  // actually accepts, so the panel renders fields like height/width/
  // negative_prompt only when the loaded architecture supports them.
  // Re-fetched whenever the active video model changes because the daemon's
  // schema endpoint 503s until a video model's video_diffusers_server.py
  // subprocess is actually running (the GET itself triggers that subprocess
  // to spawn, but the first few requests race its startup and 503 before it
  // binds its port) — so this has to retry, not just fetch once, or the
  // params panel stays empty until something else re-triggers this effect.
  useEffect(() => {
    if (!activeVideoLoaded?.model_path) {
      setVideoSchema(null);
      return;
    }
    let cancelled = false;
    const poll = async () => {
      try {
        const res = await fetch('http://127.0.0.1:8080/v1/videos/schema');
        if (!res.ok) throw new Error('schema fetch failed');
        const schema = await res.json();
        if (!cancelled) setVideoSchema(schema);
        return;
      } catch {
        if (!cancelled) setTimeout(poll, 1500);
      }
    };
    setVideoSchema(null);
    poll();
    return () => { cancelled = true; };
  }, [activeVideoLoaded?.model_path]);
  // "prompt" has its own dedicated field above the dynamic params; "image"
  // (img2video conditioning) has its own dedicated upload control below.
  const videoSchemaParams = (videoSchema?.params || []).filter(p => p.name !== 'prompt' && p.type !== 'image_b64');
  const videoSchemaSupportsImage = !videoSchema || (videoSchema.params || []).some(p => p.type === 'image_b64');
  // Re-seed the dynamic param values whenever the schema changes.
  useEffect(() => {
    const defaults = {};
    videoSchemaParams.forEach(p => {
      defaults[p.name] = meshParamDefaultValue(p);
    });
    setVideoParamValues(defaults);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [videoSchema]);
  const setVideoParam = (name, value) => setVideoParamValues(current => ({ ...current, [name]: value }));
  const setMesh3dParam = (name, value) => setMesh3dParamValues(current => ({ ...current, [name]: value }));
  // Which required field (if any) is currently unfilled for the selected input
  // kind — mirrors the backend's run_model requirement: prompt for "text",
  // an image for "image"/"multi_image". Null means the form is ready to submit.
  const mesh3dMissingField = mesh3dInputKind === 'text'
    ? (!mesh3dPrompt.trim() ? 'prompt' : null)
    : (mesh3dImages.length === 0 ? 'image' : null);
  const startStudio = (loadedEntry, catalogModel) => {
    const task_tags = catalogModel?.task_tags ?? loadedEntry?.task_tags ?? [];
    const modelName = catalogModel?.name ?? loadedEntry.model_path?.split('\\').pop() ?? 'Local model';
    const chatId = `chat-${Date.now()}`;
    const initialMessages = [{ id: 'welcome', role: 'assistant', content: `Ready to use ${modelName}.`, telemetry: null }];
    setSelectedModelId(loadedEntry.model_id);
    setOpenStudios(current => {
      // Dedupe by model_path (not model_id) — backend model_ids carry a load
      // timestamp, so a re-loaded model gets a fresh id and would otherwise
      // create a duplicate status-strip entry for the same physical model.
      const idx = current.findIndex(studio => studio.modelPath && studio.modelPath === loadedEntry.model_path);
      const entry = { modelId: loadedEntry.model_id, name: modelName, task_tags, modelPath: loadedEntry.model_path };
      if (idx >= 0) {
        const next = [...current];
        next[idx] = entry;
        return next;
      }
      return [...current, entry];
    });
    const session = { id: chatId, title: CATEGORY_DEFAULT_TITLE[categoryForTags(task_tags)] ?? 'New chat', modelId: loadedEntry.model_id, modelPath: loadedEntry.model_path, modelName, task_tags, studioKey: canonicalStudioKey(loadedEntry.model_path, modelName), messages: initialMessages, createdAt: Date.now(), updatedAt: Date.now() };
    setChatSessions(current => [...current, session]);
    setActiveChatId(chatId);
    openModelStudio({ task_tags });
  };
  // Lets a run_model tool-call chip in the orchestrator chat jump the user
  // to the matching studio tab and mirror that job's own live progress /
  // result there, without disturbing anything about how the chip itself
  // renders. Purely additive: reads from the toolEvent already in scope,
  // observes the existing job via polling (does not start a new one).
  const expandToolCallToStudio = (event) => {
    const modelName = event.arguments?.model;
    if (!event.jobId || !modelName) { console.warn('expandToolCallToStudio: missing jobId or model name', event); return; }
    const catalogModel = detectedModels.find(m => m.name === modelName);
    const loadedModel = (catalogModel && loadedModels.find(m => m.model_path === catalogModel.path))
      ?? loadedModels.find(m => m.model_path?.endsWith(modelName));
    const task_tags = catalogModel?.task_tags ?? loadedModel?.task_tags;
    if (!task_tags) { console.warn('expandToolCallToStudio: could not resolve model', modelName); return; }
    const category = categoryForTags(task_tags);
    const sync = {
      image: { setJobId: setActiveImageJobId, setProgress: setImgProgress, setGenerating: setIsGeneratingImg, setSrc: setImgSrc, setPrompt: setImgPrompt },
      video: { setJobId: setActiveVideoJobId, setProgress: setVideoProgress, setGenerating: setIsGeneratingVideo, setSrc: setVideoSrc, setPrompt: setVideoPrompt },
      mesh3d: { setJobId: setActiveMeshJobId, setProgress: setMeshProgress, setGenerating: setIsGeneratingMesh, setSrc: (url) => setMesh3dResult(url ? { url, format: guessMeshFormat(url) } : null), setPrompt: setMesh3dPrompt },
      tts: { setJobId: setActiveTtsJobId, setProgress: setTtsProgress, setGenerating: setIsGeneratingTts, setSrc: setAudioSrc, setPrompt: setTtsInput },
    }[category];

    const tab = category === 'tts' ? 'audio' : category === 'audio' ? 'transcribe' : category;
    const resolvedModelPath = loadedModel?.model_path ?? catalogModel?.path;
    if (loadedModel) {
      setOpenStudios(current => {
        const idx = current.findIndex(s => s.modelPath && s.modelPath === loadedModel.model_path);
        const entry = { modelId: loadedModel.model_id, name: catalogModel?.name ?? modelName, task_tags, modelPath: loadedModel.model_path };
        if (idx >= 0) {
          const next = [...current];
          next[idx] = entry;
          return next;
        }
        return [...current, entry];
      });
    }
    // Mirror startStudio's session-creation pattern so the model shows up in
    // the sidebar STUDIOS list, even if loadedModels hasn't caught up yet
    // (run_model may load the model server-side before the frontend polls).
    // Also switch activeChatId to that session, same as clicking it in the
    // sidebar would — otherwise the sidebar highlight (keyed off activeChatId)
    // stays stuck on the orchestrator while the visible tab has moved on.
    if (resolvedModelPath) {
      const studioKey = canonicalStudioKey(resolvedModelPath, catalogModel?.name ?? modelName);
      const existingSession = chatSessions.find(s => s.studioKey === studioKey);
      if (existingSession) {
        setActiveChatId(existingSession.id);
      } else {
        const newSession = {
          id: `chat-${Date.now()}`,
          title: catalogModel?.name ?? modelName,
          modelId: loadedModel?.model_id,
          modelPath: resolvedModelPath,
          modelName: catalogModel?.name ?? modelName,
          task_tags,
          studioKey,
          messages: [{ id: 'welcome', role: 'assistant', content: `Expanded from the orchestrator (prompt: "${event.arguments?.prompt ?? ''}").`, telemetry: null }],
          createdAt: Date.now(),
          updatedAt: Date.now(),
        };
        setChatSessions(current => [...current, newSession]);
        setActiveChatId(newSession.id);
      }
      if (loadedModel?.model_id) setSelectedModelId(loadedModel.model_id);
    }
    setActiveTab(['chat', 'image', 'audio', 'transcribe', 'video', 'mesh3d', 'embeddings'].includes(tab) ? tab : 'chat');

    // Only image/video/mesh3d/tts have a per-job progress+result shape to
    // mirror; chat/transcribe/embeddings run_model calls just switch tabs.
    if (!sync) return;

    sync.setJobId(event.jobId);
    const running = event.status === 'executing';
    sync.setGenerating(running);
    sync.setProgress(running ? event.progress : null);
    if (event.media?.url) sync.setSrc(event.media.url);
    sync.setPrompt(event.arguments?.prompt ?? '');

    const params = event.arguments?.params || {};
    if (category === 'video') setVideoParamValues(current => ({ ...current, ...params }));
    if (category === 'mesh3d') setMesh3dParamValues(current => ({ ...current, ...params }));
    if (category === 'image') {
      if ('steps' in params) setImgSteps(params.steps);
      if ('seed' in params) setImgSeed(params.seed);
      if ('cfg_scale' in params) setImgCfgScale(params.cfg_scale);
      else if ('guidance_scale' in params) setImgCfgScale(params.guidance_scale);
      if ('width' in params) setImgWidth(params.width);
      if ('height' in params) setImgHeight(params.height);
      if ('negative_prompt' in params) setImgNegativePrompt(params.negative_prompt);
    }
    if (category === 'tts' && 'speed' in params) setTtsSpeed(params.speed);

    // Prefill the reference image for image/video/mesh3d tool calls that took
    // one as input. Only fetch when the tool call actually specified an image
    // handle — absence has server-side "fall back to most recent asset"
    // behavior we can't see from here, so we leave the field empty rather
    // than guess. Handles may come back as a bare id or a path like
    // "/v1/media/<id>"; mirror the daemon's resolve_media_handle and take
    // whatever follows the last "/".
    const mediaHandle = event.arguments?.image;
    if (mediaHandle && (category === 'image' || category === 'video' || category === 'mesh3d')) {
      const mediaId = mediaHandle.split('/').pop();
      fetch(`http://127.0.0.1:8080/v1/media/${mediaId}`)
        .then(res => (res.ok ? res.blob() : Promise.reject(new Error(`status ${res.status}`))))
        .then(blob => new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result);
          reader.onerror = reject;
          reader.readAsDataURL(blob);
        }))
        .then(dataUrl => {
          if (category === 'image') {
            setImgInitImage(dataUrl);
            setImgInitImageName('from orchestrator');
          } else if (category === 'video') {
            setVideoInitImage(dataUrl);
            setVideoInitImageName('from orchestrator');
          } else if (category === 'mesh3d') {
            setMesh3dImages([{ name: 'from orchestrator', dataUrl }]);
            setMesh3dInputKind('image');
          }
        })
        .catch(err => console.warn('expandToolCallToStudio: failed to fetch reference image', err));
    }

    if (!running) return;
    const interval = setInterval(() => {
      fetch(`http://127.0.0.1:8080/v1/jobs/${event.jobId}/progress`)
        .then(res => (res.ok ? res.json() : null))
        .then(record => {
          if (!record) return;
          const terminal = record.status === 'done' || record.status === 'error' || record.status === 'cancelled';
          sync.setProgress(terminal ? null : record);
          sync.setGenerating(!terminal);
          if (terminal) {
            if (record.media_handle) sync.setSrc(`http://127.0.0.1:8080${record.media_handle}`);
            clearInterval(interval);
          }
        })
        .catch(() => { });
    }, 200);
  };
  // Writes go to the orchestrator's own transcript (targetId === null) or to
  // one specific chatSessions entry. `targetId` is explicit because a stream
  // and its run_model job pollers outlive the click that started them: if
  // they resolved the destination lazily from activeChatId they'd follow the
  // user into whatever studio they opened next (e.g. via the tool chip's
  // Expand button) and overwrite that transcript — or, worse, write the newly
  // opened studio's messages back into the session they started from.
  const syncMessages = (updater, targetId = activeChatId) => {
    const apply = (previous) => (typeof updater === 'function' ? updater(previous) : updater);
    if (targetId) {
      setChatSessions(current => current.map(session =>
        session.id === targetId ? { ...session, messages: apply(session.messages ?? []), updatedAt: Date.now() } : session));
    } else {
      setOrchestratorMessages(apply);
    }
  };
  const generateChatTitle = async (sessionId, prompt, sessionModelPath, sessionModelId) => {
    try {
      const response = await fetch('http://127.0.0.1:8080/v1/chat/completions/stream', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        // enable_thinking: false is required here — otherwise the backend
        // defaults to reasoning mode on, and max_tokens is too small to get
        // past the <think> phase into any delta.content, leaving title empty.
        body: JSON.stringify({ model: sessionModelPath, model_id: sessionModelId, max_tokens: 16, temperature: 0.2, enable_thinking: false, enable_tools: false, messages: [{ role: 'user', content: `Summarize the message below as a short label of 3-6 words. No punctuation, no quotes, no explanation.\n\nMessage: hello\nLabel: Casual Greeting\n\nMessage: ${prompt}\nLabel:` }] }),
      });
      if (!response.body) return;
      const reader = response.body.getReader(); const decoder = new TextDecoder(); let title = '';
      while (true) { const { done, value } = await reader.read(); if (done) break; for (const line of decoder.decode(value, { stream: true }).split('\n')) { try { const raw = line.replace(/^data:\s*/, ''); if (raw && raw !== '[DONE]') title += JSON.parse(raw)?.choices?.[0]?.delta?.content ?? ''; } catch { } } }
      // Some small local models echo meta-words from the instruction (e.g.
      // "chat"/"title"/"label") or restate a "Label:"-style prefix instead of
      // just answering — strip those known leakage patterns.
      title = title.replace(/[\n#*_`]/g, ' ').trim()
        .replace(/^["'“”‘’]+|["'“”‘’]+$/g, '')
        .replace(/^(chat\s+)?title\s*:\s*/i, '')
        .replace(/^(topic|label)\s*:\s*/i, '')
        .trim()
        .replace(/^(chat|title|label)(\s+(chat|title|label))*\s+/i, '')
        .replace(/\s+(chat|title|label)(\s+(chat|title|label))*$/i, '')
        .trim()
        .slice(0, 64);
      if (title) setChatSessions(current => current.map(session => session.id === sessionId ? { ...session, title } : session));
    } catch { }
  };
  // Was `loadedModels[0] ?? null` — that leaked whichever model happened to
  // load first regardless of modality. Scoped to activeChatLoaded (chat
  // category only) so the Chat Studio strip never shows another modality's
  // model.
  const modelStatus = {
    is_loaded: Boolean(activeChatLoaded),
    model_path: activeChatLoaded?.model_path,
    gpu_layers: activeChatLoaded?.gpu_layers,
    context_size: activeChatLoaded?.context_size,
  };


  useEffect(() => {
    if (!configTarget) return;
    const card = modelCards.find(c => c.id === configTarget.cardId);
    const sizeBytes = card?.size_bytes || 0;
    if (!sizeBytes) { setModelFitPreview(null); return; }
    fetch('http://127.0.0.1:8080/v1/fit-estimator', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model_size_bytes: sizeBytes,
        context_size: configTarget.context_size ?? 4096,
        modality: categoryForTags(configTarget.task_tags),
      }),
    })
      .then(res => res.json())
      .then(data => setModelFitPreview(data))
      .catch(() => { });
  }, [configTarget?.cardId, configTarget?.context_size, configTarget?.task_tags, modelCards]);

  const messagesEndRef = useRef(null);
  const chatHistoryRef = useRef(null);
  // Tracks whether the user is currently scrolled near the bottom of the
  // chat history; used to avoid yanking the view back down while they're
  // reading scrollback during rapid streaming updates (e.g. tool calls).
  const isNearBottomRef = useRef(true);
  // Tracks the AbortController for the in-flight chat stream so a "Stop"
  // button can cancel it; cleared once the stream ends (success or abort).
  const streamAbortRef = useRef(null);

  const handleChatHistoryScroll = () => {
    const el = chatHistoryRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    isNearBottomRef.current = distanceFromBottom < 120;
  };

  useEffect(() => {
    if (isNearBottomRef.current) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [displayedMessages]);

  const sysInfoInFlight = useRef(false);
  const fetchSystemInfo = async () => {
    if (sysInfoInFlight.current) return;
    sysInfoInFlight.current = true;
    try {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 4000);
      const res = await fetch('http://127.0.0.1:8080/v1/system/info', { signal: controller.signal });
      clearTimeout(timeout);
      const data = await res.json();
      if (data) {
        setSysInfo({
          cpu_cores: data.cpu_cores || 20,
          total_ram_gb: (data.total_ram_bytes / (1024 * 1024 * 1024)).toFixed(1),
          free_ram_gb: (data.available_ram_bytes / (1024 * 1024 * 1024)).toFixed(1),
          total_vram_gb: (data.total_vram_bytes / (1024 * 1024 * 1024)).toFixed(1),
          free_vram_gb: (data.free_vram_bytes / (1024 * 1024 * 1024)).toFixed(1),
          gpu_name: data.gpu_name || null
        });
      }
    } catch { } finally {
      sysInfoInFlight.current = false;
    }
  };

  useEffect(() => {
    fetchSystemInfo();
    const interval = setInterval(fetchSystemInfo, 3000);
    const onVisibility = () => { if (document.visibilityState === 'visible') fetchSystemInfo(); };
    document.addEventListener('visibilitychange', onVisibility);
    return () => { clearInterval(interval); document.removeEventListener('visibilitychange', onVisibility); };
  }, []);

  // Shared fetch+SSE-parsing core, reused by both a fresh user send and a
  // resumed request after the user picks a model from a model_choice prompt.
  const streamAssistantResponse = async (requestBody, aiId, titleText, priorContent = '') => {
    // Pinned once: every write this request makes (including the job pollers
    // below, which keep running after the user navigates away) belongs to the
    // conversation that started it, not to whichever one is on screen later.
    const targetChatId = activeChatId;
    // Accumulators for the two phases. When resuming into an existing
    // assistant message (e.g. after a model_choice pick), seed these from
    // that message's current content so the earlier reasoning/tool call
    // aren't wiped out by the first updateMsg() of this new hop.
    const seeded = parseThinking(priorContent);
    let thinkingBuf = seeded.thinking;
    let answerBuf = seeded.answer;
    let inThinkingPhase = true; // reasoning_content comes before content
    let tokenCount = 0;
    let finishReason = null;
    let gotModelChoice = false;
    const startTime = Date.now();
    // Polling intervals for delegated tool jobs (run_model), keyed by job_id.
    // A job's real lifecycle (running/done/error/cancelled) is only known via
    // this poll of /v1/jobs/:id/progress — the tool_result SSE event for
    // run_model fires almost instantly now (the daemon just started the
    // background job) and must NOT be treated as job completion. Intervals
    // are cleared solely by the poller itself reaching a terminal status;
    // they intentionally outlive this streaming request so a still-running
    // background job keeps updating its toolEvent after the LLM's own text
    // response has finished streaming.
    const jobIntervals = new Map();
    const stopJobPolling = (jobId) => {
      const interval = jobIntervals.get(jobId);
      if (interval) {
        clearInterval(interval);
        jobIntervals.delete(jobId);
      }
    };

    // textDelta is the piece of newly-arrived answer text (if any) for this
    // update. It's appended to the last block in the message's ordered
    // `blocks` list if that block is text, otherwise it starts a new text
    // block — this is what keeps text and tool_call/tool_result blocks in
    // the chronological order they actually streamed in, instead of the
    // old approach of accumulating all text separately from all tool events.
    const updateMsg = (done = false, textDelta = null) => {
      const elapsed = (Date.now() - startTime) / 1000;
      const speed = elapsed > 0.1 ? (tokenCount / elapsed).toFixed(1) : '0.0';
      const telemetry = `${speed} tok/s | CUDA GPU`;

      // Compose the combined content string that parseThinking() can parse
      // (used only to recover the <think> block for display, and to seed
      // the next hop's thinkingBuf/answerBuf on a model_choice resume).
      const combined = thinkingBuf
        ? `<think>\n${thinkingBuf}\n</think>\n\n${answerBuf}`
        : answerBuf;
      syncMessages(prev =>
        prev.map(m => {
          if (m.id !== aiId) return m;
          let blocks = m.blocks || [];
          if (textDelta) {
            const last = blocks[blocks.length - 1];
            blocks = last && last.type === 'text'
              ? [...blocks.slice(0, -1), { ...last, text: last.text + textDelta }]
              : [...blocks, { type: 'text', text: textDelta }];
          }
          if (done) {
            blocks = blocks.filter(b => b.type !== 'tool' || b.status !== 'executing' || jobIntervals.has(b.jobId));
          }
          return { ...m, content: combined, loading: !done, telemetry: telemetry, truncated: done && finishReason === 'length', blocks };
        }),
        targetChatId
      );
    };

    const controller = new AbortController();
    streamAbortRef.current = controller;

    try {
      const res = await fetch('http://127.0.0.1:8080/v1/chat/completions/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
        signal: controller.signal,
      });

      if (!res.ok || !res.body) {
        const bodyText = await res.text().catch(() => '');
        throw new Error(bodyText || `HTTP ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let sseBuffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // SSE lines look like: data: {...json...}\n — a read can end mid-line,
        // so keep the trailing partial in a buffer instead of dropping it.
        sseBuffer += decoder.decode(value, { stream: true });
        const lines = sseBuffer.split('\n');
        sseBuffer = lines.pop() ?? '';
        for (const line of lines) {
          const raw = line.startsWith('data:') ? line.slice(5).trim() : line.trim();
          if (!raw || raw === '[DONE]') continue;

          try {
            const chunk = JSON.parse(raw);

            if (chunk.type === 'model_choice') {
              // Keep the assistant bubble — it holds the reasoning and the
              // list_models tool call. Just stop its spinner and remember
              // which message to continue into once the user picks a model.
              gotModelChoice = true;
              syncMessages(prev => prev.map(m => m.id === aiId ? { ...m, loading: false } : m), targetChatId);
              setPendingModelChoice({ options: chunk.options, requestBody, aiId });
              continue;
            }

            if (chunk.type === 'error') {
              syncMessages(prev => prev.map(m =>
                m.id === aiId ? { ...m, content: 'Error: ' + chunk.message, loading: false } : m
              ), targetChatId);
              continue;
            }

            if (chunk.type === 'tool_status') {
              const jobId = chunk.job_id || null;
              syncMessages(prev => prev.map(m =>
                m.id === aiId
                  ? { ...m, blocks: [...(m.blocks || []), { type: 'tool', name: chunk.tool_name, status: 'executing', jobId, progress: null, arguments: chunk.arguments || null }] }
                  : m
              ), targetChatId);
              if (jobId) {
                const interval = setInterval(() => {
                  fetch(`http://127.0.0.1:8080/v1/jobs/${jobId}/progress`)
                    .then(res => (res.ok ? res.json() : null))
                    .then(record => {
                      if (!record) return;
                      const terminal = record.status === 'done' || record.status === 'error' || record.status === 'cancelled';
                      syncMessages(prev => prev.map(m => {
                        if (m.id !== aiId) return m;
                        const blocks = (m.blocks || []).map(b => {
                          if (b.type !== 'tool' || b.jobId !== jobId) return b;
                          const media = record.media_handle ? {
                            url: `http://127.0.0.1:8080${record.media_handle}`,
                            type: record.media_type || 'application/octet-stream',
                            content: '',
                          } : b.media || null;
                          return {
                            ...b,
                            progress: terminal ? null : record,
                            status: terminal ? record.status : b.status,
                            detail: record.status === 'error' ? (record.message || b.detail) : b.detail,
                            media,
                          };
                        });
                        return { ...m, blocks };
                      }), targetChatId);
                      if (terminal) stopJobPolling(jobId);
                    })
                    .catch(() => { });
                }, 200);
                jobIntervals.set(jobId, interval);
              }
              continue;
            }

            if (chunk.type === 'tool_result') {
              const jobId = chunk.job_id || null;

              // job_id is stamped on every tool call's SSE events, not just
              // run_model's — but only run_model's media-generation jobs
              // actually register a progress record (job_id ties into the
              // shared job_progress map only there; text-model run_model
              // calls, list_models, get_generation_progress, cancel_job etc.
              // resolve synchronously with no such record). Check the
              // record's existence rather than trusting job_id/tool_name
              // alone: if it exists, this is a real background job — leave
              // its toolEvent driven by the polling interval already started
              // from tool_status and do not resolve it here.
              let hasActiveJob = false;
              if (jobId) {
                try {
                  const progRes = await fetch(`http://127.0.0.1:8080/v1/jobs/${jobId}/progress`);
                  hasActiveJob = progRes.ok;
                } catch {
                  hasActiveJob = false;
                }
              }

              if (hasActiveJob) {
                syncMessages(prev => prev.map(m => {
                  if (m.id !== aiId) return m;
                  const blocks = (m.blocks || []).map(b => (b.type === 'tool' && b.jobId === jobId && b.status === 'executing'
                    ? { ...b, arguments: chunk.arguments || b.arguments }
                    : b));
                  return { ...m, blocks };
                }), targetChatId);
                continue;
              }

              if (jobId) stopJobPolling(jobId);

              const mediaResult = chunk.media_url ? {
                url: `http://127.0.0.1:8080${chunk.media_url}`,
                type: chunk.media_type || 'application/octet-stream',
                content: chunk.content || '',
              } : null;

              syncMessages(prev => prev.map(m => {
                if (m.id !== aiId) return m;
                // Close out the matching in-flight call rather than appending a
                // second entry — one tool_status pairs with one tool_result.
                const blocks = [...(m.blocks || [])];
                const idx = blocks.map(b => (b.type === 'tool' ? b.status : null)).lastIndexOf('executing');
                const resolved = {
                  type: 'tool',
                  name: chunk.tool_name,
                  status: chunk.is_error ? 'error' : 'done',
                  detail: chunk.is_error ? chunk.content : null,
                  jobId: jobId,
                  progress: null,
                  media: mediaResult,
                  arguments: chunk.arguments || (idx >= 0 ? blocks[idx].arguments : null) || null,
                };
                if (idx >= 0) blocks[idx] = resolved; else blocks.push(resolved);

                return { ...m, blocks };
              }), targetChatId);
              continue;
            }

            const choice = chunk?.choices?.[0] ?? {};
            const delta = choice.delta ?? {};
            if (choice.finish_reason) finishReason = choice.finish_reason;

            let chunkAdded = false;
            let textDelta = null;
            if (delta.reasoning_content != null) {
              thinkingBuf += delta.reasoning_content;
              inThinkingPhase = true;
              tokenCount++;
              chunkAdded = true;
            }
            if (delta.content != null && delta.content !== '') {
              if (inThinkingPhase) inThinkingPhase = false;
              answerBuf += delta.content;
              textDelta = delta.content;
              tokenCount++;
              chunkAdded = true;
            }
            if (chunkAdded) {
              updateMsg(false, textDelta);
            }
          } catch {
            // partial JSON chunk, skip
          }
        }
      }

      if (gotModelChoice) return;

      // Stream finished — mark done
      updateMsg(true);
      const targetSession = chatSessions.find(item => item.id === targetChatId);
      if (titleText && targetChatId && targetSession?.title === 'New chat') {
        generateChatTitle(targetChatId, titleText, targetSession.modelPath, targetSession.modelId);
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        // User-initiated stop — keep whatever partial content streamed in,
        // just drop the loading spinner instead of showing an error.
        updateMsg(true);
      } else {
        syncMessages(prev =>
          prev.map(m =>
            m.id === aiId
              ? { ...m, content: 'Error: ' + err.message, loading: false }
              : m
          ),
          targetChatId
        );
      }
    } finally {
      // Deliberately do NOT clear jobIntervals here: run_model's background
      // jobs (image/video/mesh/tts) keep running on the daemon after this
      // hop's text stream ends, and the progress poller is the only thing
      // that will ever mark those tool blocks done/error and attach their
      // media. Each interval stops itself once its job reaches a terminal
      // status (see the tool_status handler above).
      setIsGenerating(false);
      if (streamAbortRef.current === controller) streamAbortRef.current = null;
    }
  };

  // Polls GET /v1/model/list until the target model's backend process reports
  // "ready" (or "error"/timeout). Model loading spawns the backend process
  // asynchronously on the daemon, so a message sent right after loading a
  // model can otherwise race ahead of llama-server actually being ready to
  // serve requests and silently produce no reply. Entries without a `status`
  // field (older daemon builds, or non-chat modalities) are treated as ready.
  const waitForModelReady = async (modelId, onWaiting) => {
    if (!modelId) return { ready: true };
    const deadline = Date.now() + 120000;
    while (Date.now() < deadline) {
      try {
        const res = await fetch('http://127.0.0.1:8080/v1/model/list');
        const data = await res.json();
        const entry = Array.isArray(data) ? data.find(m => m.model_id === modelId) : null;
        if (!entry || !entry.status || entry.status === 'ready') return { ready: true };
        if (entry.status === 'error') return { ready: false, error: entry.status_message || 'Model failed to load.' };
        onWaiting?.();
      } catch {
        // network hiccup — keep polling until the deadline
      }
      await new Promise(resolve => setTimeout(resolve, 600));
    }
    return { ready: false, error: 'Timed out waiting for the model to finish loading.' };
  };

  // Appends a fresh user message + placeholder assistant message and streams
  // the response. Shared by the normal composer submit path and the
  // edit-and-resend path — both just need to get a user turn onto the wire
  // starting from whatever messages are already in the session.
  const sendTurn = async (userText, msgAttachments) => {
    const aiId = 'ai-' + Date.now();
    const imageAttachments = msgAttachments.filter(a => a.type?.startsWith('image/'));
    const userMsg = { id: 'u-' + Date.now(), role: 'user', content: userText, attachments: msgAttachments };
    const tempAiMsg = { id: aiId, role: 'assistant', content: '', loading: true, blocks: [] };

    const priorMessages = chatSessions.find(s => s.id === activeChatId)?.messages ?? [];
    const history = priorMessages
      .filter(m => m.role === 'user' || m.role === 'assistant')
      .map(m => {
        let content = m.role === 'assistant' ? (parseThinking(m.content).answer || m.content) : m.content;
        // Always record that a tool ran, regardless of whether the model also
        // wrote trailing text — covers both the normal case and the model
        // saying nothing after the call (or the user interrupting before it did).
        if (m.role === 'assistant' && m.blocks && m.blocks.length > 0) {
          const toolTrace = '[used tool: ' + m.blocks.map(b => b.name).join(', ') + ']';
          content = content && content.trim() ? `${content}\n${toolTrace}` : toolTrace;
        }
        return { role: m.role, content };
      })
      .filter(m => m.content && m.content.trim());

    syncMessages(prev => [...prev, userMsg, tempAiMsg]);
    setIsGenerating(true);

    const targetModelId = activeChatId ? selectedModelId : orchestratorModelId;
    const { ready, error } = await waitForModelReady(targetModelId, () => {
      syncMessages(prev => prev.map(m => m.id === aiId ? { ...m, content: 'Model is still loading…' } : m));
    });
    if (!ready) {
      syncMessages(prev => prev.map(m => m.id === aiId ? { ...m, loading: false, content: error } : m));
      setIsGenerating(false);
      return;
    }

    const requestBody = {
      model: modelPath,
      model_id: targetModelId,
      conversation_id: activeChatId,
      messages: [...history, {
        role: 'user', content: imageAttachments.length ? [
          { type: 'text', text: userText },
          ...imageAttachments.map(attachment => ({ type: 'image_url', image_url: { url: attachment.dataUrl } })),
        ] : userText
      }],
      max_tokens: 4096,
      temperature: parseFloat(temperature),
      top_p: parseFloat(topP),
      system_prompt: systemPrompt || '',
      autopilot,
      enable_thinking: reasoningEnabled,
    };

    await streamAssistantResponse(requestBody, aiId, userText);
  };

  const handleSendMessage = async () => {
    if ((!inputPrompt.trim() && attachments.length === 0) || isGenerating) return;

    const userText = inputPrompt.trim();
    const msgAttachments = attachments;
    setInputPrompt('');
    setAttachments([]);

    await sendTurn(userText, msgAttachments);
  };

  // Truncates the active session's messages back to (not including) the
  // edited message, clears the conversation's generated-assets bucket on the
  // backend (assets aren't tracked per-turn, so we can't selectively keep
  // pre-edit ones), and resends the edited text as a fresh turn — standard
  // "edit and regenerate branch" behavior.
  const handleEditAndResend = async (msgId, newText, msgAttachments) => {
    if (!activeChatId || isGenerating || !newText.trim()) return;

    setEditingMsgId(null);

    const session = chatSessions.find(item => item.id === activeChatId);
    const idx = (session?.messages ?? []).findIndex(m => m.id === msgId);
    if (idx === -1) return;

    syncMessages(prev => prev.slice(0, idx));
    fetch(`http://127.0.0.1:8080/v1/conversations/${activeChatId}/reset-assets`, { method: 'POST' }).catch(() => { });

    await sendTurn(newText.trim(), msgAttachments ?? []);
  };

  const cancelEditingMessage = () => {
    setEditingMsgId(null);
    setInputPrompt('');
    setAttachments([]);
  };

  // Shared submit handler for the composer's Enter key and Send button —
  // branches into the edit-and-resend flow when a message is being edited,
  // otherwise behaves exactly like the normal send.
  const handleComposerSubmit = async () => {
    if (editingMsgId) {
      const msgId = editingMsgId;
      const newText = inputPrompt.trim();
      const msgAttachments = attachments;
      setInputPrompt('');
      setAttachments([]);
      await handleEditAndResend(msgId, newText, msgAttachments);
      return;
    }
    await handleSendMessage();
  };

  // Aborts the in-flight chat stream fetch. This stops the UI from waiting
  // on further tokens immediately; whether the backend model keeps computing
  // after the client disconnects depends on the daemon/llama-server's own
  // handling of a dropped connection.
  const handleStopGeneration = () => {
    streamAbortRef.current?.abort();
  };

  const handlePickModel = async (option) => {
    if (!pendingModelChoice) return;
    const { requestBody, aiId } = pendingModelChoice;
    setPendingModelChoice(null);

    // Continue streaming the chosen model's result INTO the same assistant
    // message so the earlier reasoning and tool call stay visible.
    const priorContent = displayedMessages.find(m => m.id === aiId)?.content || '';
    syncMessages(prev => prev.map(m => m.id === aiId ? { ...m, loading: true } : m));
    setIsGenerating(true);

    await streamAssistantResponse({ ...requestBody, forced_model: option.model }, aiId, null, priorContent);
  };

  const addAttachments = async (files) => {
    // Images require an image-input backend and feed the model; mesh files are
    // preview-only (rendered in the viewer, never sent as model input).
    const picked = Array.from(files).filter(file =>
      (acceptsImageInput && file.type.startsWith('image/')) || isMeshResource(file.type, file.name)
    );
    const accepted = await Promise.all(picked.slice(0, 4).map(file => new Promise(resolve => {
      const reader = new FileReader();
      reader.onload = () => resolve({ name: file.name, type: file.type, dataUrl: reader.result });
      reader.readAsDataURL(file);
    })));
    setAttachments(current => [...current, ...accepted]);
  };

  // POSTs a cancel request for an in-flight job. The backend flags the job;
  // the existing progress poll picks up status "cancelled" once it notices.
  const cancelGenerationJob = (jobId) => {
    if (!jobId) return;
    fetch(`http://127.0.0.1:8080/v1/jobs/${jobId}/cancel`, { method: 'POST' }).catch(() => { });
  };

  // Fires a generation request while polling GET /v1/jobs/:id/progress every
  // ~200ms on a separate timer (no await in between), so the progress bar
  // updates concurrently with the non-streaming generation fetch. `setJobId`
  // (optional) exposes the generated job id to the caller for the duration of
  // the job, e.g. so a Cancel button can target it.
  const runGenerationJob = async (url, body, setProgress, setJobId) => {
    const jobId = crypto.randomUUID();
    if (setJobId) setJobId(jobId);
    let cancelled = false;
    const fetchPromise = fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...body, job_id: jobId }),
    });
    const interval = setInterval(() => {
      fetch(`http://127.0.0.1:8080/v1/jobs/${jobId}/progress`)
        .then(res => (res.ok ? res.json() : null))
        .then(record => {
          if (!record) return;
          setProgress(record);
          if (record.status === 'cancelled') cancelled = true;
        })
        .catch(() => { });
    }, 200);
    try {
      const res = await fetchPromise;
      if (!res.ok) {
        const text = (await res.text()).trim();
        let message = text;
        let missingComponents = null;
        let envNotInstalled = null;
        try {
          const parsed = JSON.parse(text);
          message = parsed?.error || parsed?.message || text;
          if (Array.isArray(parsed?.missing_components) && parsed.missing_components.length > 0) {
            missingComponents = parsed.missing_components;
          }
          if (parsed?.envNotInstalled) envNotInstalled = parsed.envNotInstalled;
        } catch {
          // plain-text error body, use as-is
        }
        const err = new Error(cancelled ? 'Generation cancelled' : message);
        err.cancelled = cancelled;
        if (missingComponents) err.missingComponents = missingComponents;
        if (envNotInstalled) err.envNotInstalled = envNotInstalled;
        throw err;
      }
      return await res.json();
    } catch (e) {
      if (cancelled) e.cancelled = true;
      throw e;
    } finally {
      clearInterval(interval);
      setProgress(null);
      if (setJobId) setJobId(null);
    }
  };

  const handleGenerateImage = async () => {
    setIsGeneratingImg(true);
    setImgMissingComponents(null);
    try {
      const data = await runGenerationJob('http://127.0.0.1:8080/v1/images/generations', {
        prompt: imgPrompt,
        negative_prompt: imgNegativePrompt,
        steps: Number(imgSteps),
        cfg_scale: Number(imgCfgScale),
        width: Number(imgWidth),
        height: Number(imgHeight),
        seed: Number(imgSeed),
        image: imgInitImage ? imgInitImage.split(',')[1] : null,
        strength: Number(imgStrength),
      }, setImgProgress, setActiveImageJobId);
      if (data?.data?.[0]?.b64_json) {
        setImgSrc('data:image/png;base64,' + data.data[0].b64_json);
        setImgMissingComponents(null);
      }
    } catch (e) {
      if (e.cancelled) {
        pushError('Generation cancelled', 'info');
      } else if (e.envNotInstalled) {
        installEnvAndRetry(e.envNotInstalled, () => handleGenerateImage());
      } else {
        if (Array.isArray(e.missingComponents) && e.missingComponents.length > 0) {
          setImgMissingComponents(e.missingComponents);
        }
        pushError('Image generation error: ' + e);
      }
    } finally {
      setIsGeneratingImg(false);
    }
  };

  const handleGenerateVideo = async () => {
    setIsGeneratingVideo(true);
    try {
      const data = await runGenerationJob('http://127.0.0.1:8080/v1/videos/generations', {
        prompt: videoPrompt,
        image_b64: videoInitImage ? videoInitImage.split(',')[1] : null,
        ...videoParamValues,
      }, setVideoProgress, setActiveVideoJobId);
      if (data?.video_b64) {
        setVideoSrc('data:video/mp4;base64,' + data.video_b64);
      }
    } catch (e) {
      if (e.cancelled) {
        pushError('Generation cancelled', 'info');
      } else if (e.envNotInstalled) {
        installEnvAndRetry(e.envNotInstalled, () => handleGenerateVideo());
      } else {
        pushError('Video generation error: ' + e);
      }
    } finally {
      setIsGeneratingVideo(false);
    }
  };

  const handleDownloadImage = async () => {
    if (!imgSrc) return;
    const blob = await (await fetch(imgSrc)).blob();
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `image-${Date.now()}.png`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const handleDownloadMedia = async (url, type) => {
    if (!url) return;
    const blob = await (await fetch(url)).blob();
    const objectUrl = URL.createObjectURL(blob);
    const ext = type?.split('/')[1]?.split(';')[0] || 'bin';
    const link = document.createElement('a');
    link.href = objectUrl;
    link.download = `generated-${Date.now()}.${ext}`;
    link.click();
    URL.revokeObjectURL(objectUrl);
  };

  const handleImgInitImage = (files) => {
    const file = files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      setImgInitImage(reader.result);
      setImgInitImageName(file.name);
    };
    reader.readAsDataURL(file);
  };

  const handleVideoInitImage = (files) => {
    const file = files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      setVideoInitImage(reader.result);
      setVideoInitImageName(file.name);
    };
    reader.readAsDataURL(file);
  };

  const handleMesh3dSingleImage = (files) => {
    const file = files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => setMesh3dImages([{ name: file.name, dataUrl: reader.result }]);
    reader.readAsDataURL(file);
  };

  const handleMesh3dMultiImages = async (files) => {
    const accepted = await Promise.all(Array.from(files).map(file => new Promise(resolve => {
      const reader = new FileReader();
      reader.onload = () => resolve({ name: file.name, dataUrl: reader.result });
      reader.readAsDataURL(file);
    })));
    setMesh3dImages(current => [...current, ...accepted]);
  };

  const handleGenerateMesh3d = async () => {
    setIsGeneratingMesh(true);
    try {
      const body = {
        prompt: mesh3dPrompt,
        input_kind: mesh3dInputKind,
        output_format: mesh3dFormat,
        texture: mesh3dTexture,
        ...mesh3dParamValues,
      };
      if (mesh3dInputKind !== 'text') {
        body.images = mesh3dImages.map(img => img.dataUrl.split(',')[1]);
      }
      const data = await runGenerationJob('http://127.0.0.1:8080/v1/models3d/generations', body, setMeshProgress, setActiveMeshJobId);
      if (data?.mesh_base64) {
        setMesh3dResult({ base64: data.mesh_base64, format: data.format || mesh3dFormat });
      }
    } catch (e) {
      if (e.cancelled) {
        pushError('Generation cancelled', 'info');
      } else if (e.envNotInstalled) {
        installEnvAndRetry(e.envNotInstalled, () => handleGenerateMesh3d());
      } else {
        pushError('3D mesh generation error: ' + e);
      }
    } finally {
      setIsGeneratingMesh(false);
    }
  };

  const handleSynthesizeSpeech = async () => {
    setIsGeneratingTts(true);
    try {
      const data = await runGenerationJob('http://127.0.0.1:8080/v1/audio/speech', {
        model: 'kokoro', input: ttsInput, speed: Number(ttsSpeed),
      }, setTtsProgress, setActiveTtsJobId);
      if (data?.audio_b64) {
        setAudioSrc('data:audio/wav;base64,' + data.audio_b64);
      }
    } catch (e) {
      if (e.cancelled) {
        pushError('Generation cancelled', 'info');
      } else if (e.envNotInstalled) {
        installEnvAndRetry(e.envNotInstalled, () => handleSynthesizeSpeech());
      } else {
        pushError('TTS error: ' + e);
      }
    } finally {
      setIsGeneratingTts(false);
    }
  };

  const handleTranscribeAudio = (files) => {
    const file = files?.[0];
    if (!file) return;
    setAsrFileName(file.name);
    setAsrText('');
    const reader = new FileReader();
    reader.onload = async () => {
      const audioB64 = reader.result.split(',')[1];
      setIsTranscribing(true);
      try {
        const response = await fetch('http://127.0.0.1:8080/v1/audio/transcriptions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ audio_b64: audioB64, language: asrLanguage || undefined }),
        });
        if (!response.ok) throw new Error(await response.text());
        const data = await response.json();
        setAsrText(data?.text || '');
      } catch (e) {
        pushError('Transcription error: ' + e);
      } finally {
        setIsTranscribing(false);
      }
    };
    reader.readAsDataURL(file);
  };

  const handleGenerateEmbeddings = async () => {
    const inputs = embedInput.split('\n').map(s => s.trim()).filter(Boolean);
    if (inputs.length === 0) return;
    setIsEmbedding(true);
    try {
      const response = await fetch('http://127.0.0.1:8080/v1/embeddings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: embedModelId || 'local', input: inputs }),
      });
      if (!response.ok) throw new Error(await response.text());
      setEmbedResults(await response.json());
    } catch (e) {
      pushError('Embeddings error: ' + e);
    } finally {
      setIsEmbedding(false);
    }
  };

  const totalRam = parseFloat(sysInfo.total_ram_gb) || 16.0;
  const freeRam = parseFloat(sysInfo.free_ram_gb) || 10.0;
  const usedRam = totalRam - freeRam;
  const ramPct = Math.min(100, Math.max(0, (usedRam / totalRam) * 100)).toFixed(0);

  const totalVram = parseFloat(sysInfo.total_vram_gb) || 8.0;
  const freeVram = parseFloat(sysInfo.free_vram_gb) || 6.0;
  const usedVram = totalVram - freeVram;
  const vramPct = Math.min(100, Math.max(0, (usedVram / totalVram) * 100)).toFixed(0);
  const gpuRaw = sysInfo.gpu_name === 'NVIDIA GPU (CUDA)' ? '' : (sysInfo.gpu_name || '');
  const gpuLabel = gpuRaw.replace(/^NVIDIA\s+/i, '').trim();

  const activeDownloadCount = Object.values(hfDownloads).filter(d => d.status === 'downloading').length;
  const sortedDownloads = Object.entries(hfDownloads).sort(([, a], [, b]) => {
    const rank = s => s === 'downloading' ? 0 : 1;
    return rank(a.status) - rank(b.status);
  });

  return (
    <div className="desktop-app">
      {/* Top Header Bar */}
      <header className="desktop-header">
        <div className="logo-group">
          <img className="logo-icon" src={disposLogo} alt="Dispos Studio" />
          <div className="logo-title">Dispos Studio</div>
        </div>

        <div className="header-stats">
          <div className="stat-item">
            <Cpu size={14} style={{ color: 'var(--accent-blue)' }} />
            <span>VRAM:</span>
            <div className="stat-progress">
              <div className="stat-bar" style={{ width: `${vramPct}%` }}></div>
            </div>
            <span>{usedVram.toFixed(1)} / {totalVram.toFixed(1)} GB</span>
          </div>

          <div className="stat-item">
            <HardDrive size={14} style={{ color: 'var(--accent-purple)' }} />
            <span>RAM:</span>
            <div className="stat-progress">
              <div className="stat-bar" style={{ width: `${ramPct}%`, background: 'var(--accent-purple)' }}></div>
            </div>
            <span>{usedRam.toFixed(1)} / {totalRam.toFixed(1)} GB</span>
          </div>

          <div className="status-badge" title={sysInfo.gpu_name || 'CUDA GPU Active'}>
            <div className="status-dot"></div>
            <span>CUDA GPU Active</span>
            {gpuLabel && <span className="status-badge-gpu">{gpuLabel}</span>}
          </div>

          <button className="downloads-tracker-btn" onClick={() => setShowDownloadsPanel(current => !current)} title="Downloads">
            <Download size={14} />
            <span>Downloads</span>
            {activeDownloadCount > 0 && <span className="downloads-tracker-badge">{activeDownloadCount}</span>}
          </button>
        </div>
      </header>

      {showDownloadsPanel && (
        <>
          <div className="downloads-tracker-backdrop" onClick={() => setShowDownloadsPanel(false)} />
          <div className="downloads-tracker-panel">
            <div className="modal-header">
              <h3>Downloads</h3>
              <button className="config-sidebar-close" onClick={() => setShowDownloadsPanel(false)} title="Close">
                <X size={16} />
              </button>
            </div>
            <div className="downloads-tracker-list">
              {sortedDownloads.length === 0 && (
                <div className="hf-empty-state">
                  <Download size={22} />
                  <p>No downloads yet.</p>
                </div>
              )}
              {sortedDownloads.map(([key, download]) => {
                const hasTotal = download.total_bytes != null && download.total_bytes > 0;
                const progressPct = hasTotal ? Math.min(100, (download.downloaded_bytes / download.total_bytes) * 100) : 0;
                return (
                  <div key={key} className="hf-file-row">
                    <div className="hf-file-info">
                      <div className="hf-file-name-row">
                        <span className="hf-file-name">{download.filename}</span>
                      </div>
                      {download.repo && <span className="modal-list-item-meta">{download.repo}</span>}
                      {hasTotal ? (
                        <>
                          <div className="hf-progress-bar">
                            <div className="hf-progress-fill" style={{ width: `${progressPct}%` }} />
                          </div>
                          <span className="hf-progress-text">
                            {formatFileSize(download.downloaded_bytes)} / {formatFileSize(download.total_bytes)}
                          </span>
                        </>
                      ) : (
                        download.status === 'downloading' && (
                          <span className="hf-progress-text">{formatFileSize(download.downloaded_bytes)} downloaded</span>
                        )
                      )}
                      {download.status === 'error' && (
                        <span className="modal-list-item-meta" style={{ color: '#ef4444' }}>
                          Failed{download.error ? `: ${download.error}` : ''}
                          {download.error && (download.error.includes('403') || download.error.toLowerCase().includes('forbidden')) && (
                            <span style={{ display: 'block', marginTop: 2, fontSize: 11 }}>
                              License acceptance required —{' '}
                              <a href={`https://huggingface.co/${download.repo}`} target="_blank" rel="noreferrer" style={{ color: '#60a5fa', textDecoration: 'underline' }}>open model page</a>
                              {' '}to accept terms, then retry.
                            </span>
                          )}
                          {download.error && (download.error.includes('401') || download.error.toLowerCase().includes('unauthorized')) && (
                            <span style={{ display: 'block', marginTop: 2, fontSize: 11 }}>
                              Authentication required — add your HF token in{' '}
                              <a href="#" onClick={e => { e.preventDefault(); setActiveTab('settings'); }} style={{ color: '#60a5fa', textDecoration: 'underline' }}>General Settings</a>
                            </span>
                          )}
                        </span>
                      )}
                      {download.status === 'complete' && (
                        <span className="modal-list-item-meta" style={{ color: 'var(--accent-green)' }}>Complete</span>
                      )}
                      {download.status === 'cancelled' && (
                        <span className="modal-list-item-meta">Cancelled</span>
                      )}
                    </div>
                    <div className="hf-file-actions">
                      {download.status === 'downloading' && (
                        <button className="hf-cancel-btn" onClick={() => cancelHfDownload(download.repo, download.filename)} title="Cancel download">
                          <X size={14} />
                        </button>
                      )}
                      {download.repo && (
                        <button className="hf-download-btn" onClick={() => jumpToCatalogEntry(download.repo, download.filename)}>
                          Find in catalog
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </>
      )}

      {/* Main Content Body */}
      <div className="app-container">
        {/* Left Sidebar */}
        <div className={`sidebar ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
          <button
            type="button"
            className="sidebar-fold-toggle"
            onClick={() => setSidebarCollapsed(current => !current)}
            title={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {sidebarCollapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
          </button>

          <div className="nav-section">MODEL CENTER</div>

          <button
            className={`nav-btn ${activeTab === 'models' ? 'active' : ''}`}
            onClick={() => setActiveTab('models')}
            title="Models"
          >
            <Box size={17} /> <span className="nav-label-text">Models</span>
          </button>

          <button
            className={`nav-btn ${activeTab === 'settings' ? 'active' : ''}`}
            onClick={() => setActiveTab('settings')}
            title="General Settings"
          >
            <Settings size={17} /> <span className="nav-label-text">General Settings</span>
          </button>

          <div className="nav-section" style={{ marginTop: '1rem' }}>STUDIOS</div>
          {chatSessions.length === 0 ? <div className="sidebar-empty-hint" style={{ color: 'var(--text-secondary)', fontSize: '0.78rem', padding: '0.25rem 0.75rem 0.75rem' }}>Start a studio from a loaded model in Model Center.</div>
            : Object.entries(chatSessions.reduce((groups, chat) => {
              const modelKey = chat.studioKey || canonicalStudioKey(chat.modelPath, chat.modelName);
              if (!modelKey) return groups;
              groups[modelKey] ??= { name: chat.modelName, modelId: chat.modelId, modelPath: chat.modelPath, chats: [] };
              groups[modelKey].chats.push(chat);
              return groups;
            }, {}))
              .sort(([, a], [, b]) =>
                Math.max(...b.chats.map(chatLastModified), 0) - Math.max(...a.chats.map(chatLastModified), 0))
              .map(([modelKey, model]) => {
                const modelOpen = !(collapsedStudios[modelKey] ?? false);
                const isPendingDeleteModel = pendingDeleteModelKey === modelKey;
                const modelSessionIds = model.chats.map(c => c.id);
                const openChatSession = (chat) => {
                  // Backend model_ids include a load timestamp (mdl-{ts}-{port}),
                  // so a re-loaded model has a fresh id even when the file
                  // is the same. Resolve chat.modelId against the current
                  // daemon registry (by id, then by stable modelPath) so
                  // the UI shows the actually-loaded model instead of "No
                  // Model Loaded" for studios created under a previous
                  // daemon instance. Falls back to the stored id so
                  // messages still route through the daemon's port-50052
                  // fallback when nothing matches.
                  const liveMatch = loadedModels.find(loaded => loaded.model_id === chat.modelId)
                    ?? (chat.modelPath ? loadedModels.find(loaded => loaded.model_path === chat.modelPath) : null)
                    ?? null;
                  setSelectedModelId(liveMatch?.model_id ?? chat.modelId);
                  setActiveChatId(chat.id);
                  openModelStudio(chat);
                };
                const handleStudioToggleClick = () => {
                  if (sidebarCollapsed) {
                    const latestChat = model.chats.reduce((latest, chat) =>
                      (!latest || chatLastModified(chat) >= chatLastModified(latest)) ? chat : latest, null);
                    if (latestChat) openChatSession(latestChat);
                    return;
                  }
                  setCollapsedStudios(current => ({ ...current, [modelKey]: !(current[modelKey] ?? false) }));
                };
                const loadedEntry = loadedModels.find(loaded => loaded.model_id === model.modelId)
                  ?? (model.modelPath ? loadedModels.find(loaded => loaded.model_path === model.modelPath) : null)
                  ?? null;
                const handleNewStudioClick = (e) => {
                  e.stopPropagation();
                  if (!loadedEntry) return;
                  startStudio(loadedEntry, { name: model.name, task_tags: model.chats[0]?.task_tags });
                };
                const handleDeleteModelClick = (e) => {
                  e.stopPropagation();
                  if (isPendingDeleteModel) {
                    setPendingRemoveIds(ids => {
                      const additions = modelSessionIds.filter(id => !ids.includes(id));
                      return additions.length ? [...ids, ...additions] : ids;
                    });
                    setTimeout(() => {
                      setChatSessions(current => current.filter(item => {
                        const k = item.studioKey || canonicalStudioKey(item.modelPath, item.modelName);
                        return k !== modelKey;
                      }));
                      setPendingRemoveIds(ids => ids.filter(id => !modelSessionIds.includes(id)));
                      if (activeChatId && modelSessionIds.includes(activeChatId)) {
                        setActiveChatId(null);
                      }
                      setPendingDeleteModelKey(null);
                    }, 350);
                  } else {
                    setPendingDeleteModelKey(modelKey);
                  }
                };
                return <div key={modelKey} style={{ paddingLeft: '0.55rem' }}>
                  <div className={`studio-group-row ${isPendingDeleteModel ? 'pending-delete' : ''}`}>
                    <button
                      className="nav-btn studio-group-toggle"
                      onClick={handleStudioToggleClick}
                      title={model.name}
                    >
                      {sidebarCollapsed ? (
                        <StudioAvatarIcon modelPath={model.modelPath} />
                      ) : (
                        <>{modelOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />} <span className="nav-label-text">{model.name}</span></>
                      )}
                    </button>
                    <div className="studio-group-actions">
                      {loadedEntry && (
                        <button type="button" className="chat-session-action" onClick={handleNewStudioClick} title="New studio with this model" aria-label="New studio with this model">
                          <Plus size={13} />
                        </button>
                      )}
                      <button type="button" className={`chat-session-action delete ${isPendingDeleteModel ? 'confirm' : ''}`} onClick={handleDeleteModelClick} title={isPendingDeleteModel ? 'Click again to confirm delete all studios' : 'Delete all studios for this model'} aria-label="Delete all studios for this model">
                        {isPendingDeleteModel ? <span className="confirm-label">Delete?</span> : <X size={13} />}
                      </button>
                    </div>
                  </div>
                  {modelOpen && !sidebarCollapsed && model.chats.map(chat => {
                    const isEditing = editingSessionId === chat.id;
                    const isPendingDelete = pendingDeleteId === chat.id;
                    const commitRename = () => {
                      const trimmed = editValue.trim();
                      const targetId = editingSessionId;
                      setEditingSessionId(null);
                      setEditValue('');
                      if (!trimmed || !targetId) return;
                      setChatSessions(current => current.map(item => item.id === targetId && trimmed !== item.title ? { ...item, title: trimmed } : item));
                    };
                    const cancelRename = () => {
                      setEditingSessionId(null);
                      setEditValue('');
                    };
                    const startRename = (e) => {
                      e.stopPropagation();
                      setEditValue(chat.title);
                      setEditingSessionId(chat.id);
                    };
                    const handleDeleteClick = (e) => {
                      e.stopPropagation();
                      if (isPendingDelete) {
                        // Mark the row for fade-out, then drop it from state
                        // once the CSS transition has played.
                        setPendingRemoveIds(ids => ids.includes(chat.id) ? ids : [...ids, chat.id]);
                        setTimeout(() => {
                          setChatSessions(current => current.filter(item => item.id !== chat.id));
                          setPendingRemoveIds(ids => ids.filter(id => id !== chat.id));
                          if (activeChatId === chat.id) {
                            setActiveChatId(null);
                          }
                          setPendingDeleteId(null);
                        }, 350);
                      } else {
                        setPendingDeleteId(chat.id);
                      }
                    };
                    return <div key={chat.id} className={`nav-btn chat-session-row ${activeChatId === chat.id ? 'active' : ''} ${isEditing ? 'editing' : ''} ${isPendingDelete ? 'pending-delete' : ''} ${pendingRemoveIds.includes(chat.id) ? 'fading-out' : ''}`} style={{ paddingLeft: '2.1rem', fontSize: '0.8rem' }}>
                      <button type="button" className="chat-session-main" onClick={() => openChatSession(chat)} title={chat.title}>
                        <MessageSquare size={14} />
                        {isEditing ? (
                          <input
                            type="text"
                            className="chat-session-edit-input"
                            value={editValue}
                            onChange={e => setEditValue(e.target.value)}
                            onClick={e => e.stopPropagation()}
                            onKeyDown={e => {
                              if (e.key === 'Enter') { e.preventDefault(); commitRename(); }
                              else if (e.key === 'Escape') { e.preventDefault(); cancelRename(); }
                            }}
                            onBlur={commitRename}
                            autoFocus
                            onFocus={e => e.target.select()}
                            aria-label="Rename studio"
                          />
                        ) : (
                          <span className="chat-session-title">{displayedTitles[chat.id] ?? chat.title}</span>
                        )}
                      </button>
                      <div className="chat-session-actions">
                        <button type="button" className="chat-session-action" onClick={startRename} title="Rename studio" aria-label="Rename studio">
                          <Pencil size={13} />
                        </button>
                        <button type="button" className={`chat-session-action delete ${isPendingDelete ? 'confirm' : ''}`} onClick={handleDeleteClick} title={isPendingDelete ? 'Click again to confirm delete' : 'Delete studio'} aria-label="Delete studio">
                          {isPendingDelete ? <span className="confirm-label">Delete?</span> : <X size={13} />}
                        </button>
                      </div>
                    </div>;
                  })}
                </div>;
              })}

          {false && (<>
            {/* Legacy model controls: management now lives in the Models page. */}
            <div className="nav-section" style={{ marginTop: '1rem' }}>MODEL CONTROLS</div>

            <div className="sidebar-section">
              <label className="section-label">Model Presets</label>
              <select
                className="control-select"
                value={modelPath}
                onChange={e => {
                  setModelPath(e.target.value);
                  setConfigTarget(current => ({ ...current, model_path: e.target.value }));
                }}
              >
                <option value="C:\Users\adem2\.lmstudio\models\unsloth\Qwen3.5-0.8B-GGUF\Qwen3.5-0.8B-Q8_0.gguf">
                  Qwen 3.5 0.8B (Q8_0)
                </option>
                <option value="C:\Users\adem2\.lmstudio\models\deepseek-ai\DeepSeek-R1-Distill-Qwen-1.5B-GGUF\DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf">
                  DeepSeek R1 1.5B (Q4_K_M)
                </option>
              </select>
              <input
                type="text"
                className="control-input"
                value={modelPath}
                onChange={e => {
                  setModelPath(e.target.value);
                  setConfigTarget(current => ({ ...current, model_path: e.target.value }));
                }}
                placeholder="C:\path\to\model file"
                style={{ marginTop: '0.4rem' }}
              />
            </div>

            <div className="sidebar-section">
              <div className="slider-header">
                <label className="section-label">GPU Layers</label>
                <span className="badge-value">{gpuLayers} / 99</span>
              </div>
              <input
                type="range" min="0" max="99"
                value={gpuLayers}
                onChange={e => {
                  const value = parseInt(e.target.value, 10);
                  setGpuLayers(value);
                  setConfigTarget(current => ({ ...current, gpu_layers: value }));
                }}
                className="control-slider"
              />
              <div className="slider-hint">
                {gpuLayers === 99 ? '🔥 Full GPU' : gpuLayers === 0 ? '💻 CPU Only' : `${gpuLayers} layers → VRAM`}
              </div>
            </div>

            <div className="sidebar-section">
              <label className="section-label">Context Window</label>
              <select
                className="control-select"
                value={contextSize}
                onChange={e => {
                  const value = parseInt(e.target.value, 10);
                  setContextSize(value);
                  setConfigTarget(current => ({ ...current, context_size: value }));
                }}
              >
                <option value={2048}>2048 – Minimal VRAM</option>
                <option value={4096}>4096 – Standard</option>
                <option value={8192}>8192 – Extended</option>
                <option value={16384}>16384 – Deep Reasoning</option>
              </select>
            </div>

            <div className="sidebar-section">
              <div className="slider-header">
                <label className="section-label">Temperature</label>
                <span className="badge-value">{temperature}</span>
              </div>
              <input
                type="range" min="0.0" max="1.5" step="0.05"
                value={temperature}
                onChange={e => {
                  const value = parseFloat(e.target.value);
                  setTemperature(value);
                  setConfigTarget(current => ({ ...current, temperature: value }));
                }}
                className="control-slider"
              />
              <div className="slider-header" style={{ marginTop: '0.6rem' }}>
                <label className="section-label">Top-P</label>
                <span className="badge-value">{topP}</span>
              </div>
              <input
                type="range" min="0.1" max="1.0" step="0.05"
                value={topP}
                onChange={e => {
                  const value = parseFloat(e.target.value);
                  setTopP(value);
                  setConfigTarget(current => ({ ...current, top_p: value }));
                }}
                className="control-slider"
              />
            </div>

            <div className="vram-preview-card">
              <div className="preview-title"><Cpu size={13} /> VRAM Impact</div>
              <div className="preview-grid">
                <div className="preview-item">
                  <span>Est. Required:</span>
                  <strong>{((modelFitPreview?.total_required_vram_bytes || 1.2e9) / 1e9).toFixed(2)} GB</strong>
                </div>
                <div className="preview-item">
                  <span>Fits in GPU:</span>
                  <span className="status-yes"><CheckCircle2 size={12} /> YES</span>
                </div>
              </div>
            </div>

            <div className="control-actions">
              <button className="btn-load-model" onClick={handleLoadModel} disabled={isLoadingModel}>
                <Zap size={15} /> {isLoadingModel ? 'Loading...' : modelStatus.is_loaded ? 'Reload' : 'Load to VRAM'}
              </button>
              {modelStatus.is_loaded && (
                <button className="btn-unload-model" onClick={() => handleUnloadModel(selectedModelId)} disabled={unloadingModelId === selectedModelId}>
                  <Power size={15} /> {unloadingModelId === selectedModelId ? 'Ejecting...' : 'Eject Model'}
                </button>
              )}
            </div>
          </>)}

        </div>

        {/* Main Tab Panels */}
        <main>
          {/* 1. Chat Studio */}
          {activeTab === 'chat' && (
            <div className="tab-panel chat-studio-panel">
              {/* Model Status Strip */}
              <div className="model-status-strip">
                <div className="status-info">
                  <StatusDot state={modelDotState(activeChatStudio, activeChatLoaded)} />
                  <div className="status-text">
                    <span className="model-name">
                      {modelStatus.is_loaded
                        ? modelStatus.model_path?.split('\\').pop() || 'Model Loaded'
                        : activeChatStudio?.name || 'No Model Loaded'}
                    </span>
                    <span className="model-meta">
                      {modelStatus.is_loaded
                        ? `CUDA • ${modelStatus.gpu_layers ?? gpuLayers} GPU layers • ${modelStatus.context_size ?? contextSize} ctx`
                        : activeChatStudio ? 'Not currently loaded — load it to use this studio' : 'Use the sidebar to configure and load a model'}
                    </span>
                  </div>
                </div>
                {modelStatus.is_loaded && (
                  <button className="btn-eject-mini" onClick={() => handleUnloadModel(selectedModelId)} disabled={unloadingModelId === selectedModelId}>
                    <Power size={13} /> {unloadingModelId === selectedModelId ? 'Ejecting...' : 'Eject'}
                  </button>
                )}
              </div>

              {/* Full-width chat workspace */}
              <div className="chat-workspace">
                <div className="chat-history" ref={chatHistoryRef} onScroll={handleChatHistoryScroll}>
                  {displayedMessages.map(msg => (
                    <div key={msg.id} className={`msg-row ${msg.role}`}>
                      {msg.role === 'assistant' && <div className="avatar ai">AI</div>}
                      <div className={`bubble${editingMsgId === msg.id ? ' editing' : ''}`}>
                        <div className="bubble-actions">
                          <CopyMessageButton text={msg.role === 'assistant' ? parseThinking(msg.content).answer : msg.content} />
                          {msg.role === 'user' && activeChatId && !isGenerating && editingMsgId !== msg.id && (
                            <button type="button" className="copy-msg-btn" onClick={() => {
                              setEditingMsgId(msg.id);
                              setInputPrompt(msg.content);
                              setAttachments(msg.attachments || []);
                              composerTextareaRef.current?.focus();
                            }} title="Edit message">
                              <Pencil size={13} />
                            </button>
                          )}
                        </div>
                        {editingMsgId === msg.id && <div className="bubble-editing-label">Editing…</div>}
                        {msg.role === 'assistant' ? (
                          (() => {
                            // `thinking` is still recovered from the combined
                            // content string, but the visible turn body is
                            // rendered from `blocks` — a single ordered list of
                            // text/tool_call/tool_result segments in the exact
                            // order they streamed in, so a tool call that
                            // happened between two paragraphs renders between
                            // them (both live and after reload) instead of all
                            // tool calls being dumped after all text.
                            const { thinking, answer, hadThinkTag } = parseThinking(msg.content);
                            const blocks = msg.blocks || [];
                            return (
                              <>
                                {appSettings.showThoughtProcess && <ThinkingBlock thinkText={thinking} hadThinkTag={hadThinkTag} />}
                                {blocks.length > 0 ? blocks.map((block, idx) => (
                                  <React.Fragment key={idx}>
                                    {block.type === 'text' ? (
                                      <div className="answer-content">
                                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{block.text}</ReactMarkdown>
                                      </div>
                                    ) : (
                                      <>
                                        <ToolCallChip event={block} onCancel={() => cancelGenerationJob(block.jobId)} onExpand={() => expandToolCallToStudio(block)} />
                                        {block.media && (
                                          isMeshResource(block.media.type, block.media.url) ? (
                                            <div className="tool-media-result">
                                              <Mesh3DViewer url={block.media.url} format={guessMeshFormat(block.media.url)} />
                                            </div>
                                          ) : (
                                            <div className="tool-media-result">
                                              {block.media.type?.startsWith('audio/') && (
                                                <audio controls autoPlay src={block.media.url} style={{ width: '100%' }} />
                                              )}
                                              {block.media.type?.startsWith('image/') && (
                                                <img src={block.media.url} alt="Generated" className="tool-media-image" />
                                              )}
                                              {block.media.type?.startsWith('video/') && (
                                                <VideoPlayer src={block.media.url} style={{ width: '100%' }} />
                                              )}
                                              <button type="button" className="tool-media-download-btn" onClick={() => handleDownloadMedia(block.media.url, block.media.type)} title="Download">
                                                <Download size={12} /> Download
                                              </button>
                                            </div>
                                          )
                                        )}
                                      </>
                                    )}
                                  </React.Fragment>
                                )) : (
                                  <div className="answer-content">
                                    {answer ? (
                                      // No `blocks` on this message — either it predates the
                                      // blocks refactor (persisted session) or it's a plain
                                      // content-only message like a studio "welcome" line.
                                      // Fall back to rendering msg.content directly so those
                                      // don't silently disappear.
                                      <ReactMarkdown remarkPlugins={[remarkGfm]}>{answer}</ReactMarkdown>
                                    ) : msg.loading ? (
                                      <em style={{ opacity: 0.6, fontSize: '0.85rem', color: '#94a3b8' }}>
                                        {hadThinkTag ? 'Thinking…' : 'Generating…'}
                                      </em>
                                    ) : msg.truncated ? (
                                      <em style={{ opacity: 0.6, fontSize: '0.85rem', color: '#94a3b8' }}>
                                        [Model reached output token limit during reasoning — try re-prompting]
                                      </em>
                                    ) : null}
                                  </div>
                                )}
                              </>
                            );
                          })()
                        ) : (
                          <div className="answer-content">
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                          </div>
                        )}
                        {msg.attachments?.length > 0 && <div className="chat-attachments">
                          {msg.attachments.map(attachment => <AttachmentPreview key={attachment.dataUrl} attachment={attachment} />)}
                        </div>}
                        {msg.telemetry && (
                          <div className="meta-info">
                            <span className="tag-speed"><Zap size={11} /> {msg.telemetry}</span>
                          </div>
                        )}
                      </div>
                      {msg.role === 'user' && <div className="avatar user">YOU</div>}
                    </div>
                  ))}
                  <div ref={messagesEndRef} />
                </div>

                <div className="input-card">
                  {attachments.length > 0 && <div className="composer-attachments">
                    {attachments.map(attachment => <div key={attachment.dataUrl} className="composer-attachment">
                      <AttachmentPreview attachment={attachment} />
                      <button onClick={() => setAttachments(current => current.filter(item => item.dataUrl !== attachment.dataUrl))} title={`Remove ${attachment.name}`}><X size={13} /></button>
                    </div>)}
                  </div>}
                  {pendingModelChoice && <div className="composer-model-choice">
                    <div className="composer-model-choice-header">
                      <span>Choose a model:</span>
                      <button className="btn-dismiss-choice" onClick={() => setPendingModelChoice(null)} title="Dismiss"><X size={13} /></button>
                    </div>
                    <div className="composer-model-choice-options">
                      {pendingModelChoice.options.map(option => (
                        <button key={option.model} className="model-choice-option" onClick={() => handlePickModel(option)}>
                          <span className="model-choice-name">{option.model}</span>
                          <span className="model-choice-modality">{option.task_tags?.join(', ')}{option.loaded ? ' · loaded' : ''}</span>
                        </button>
                      ))}
                    </div>
                  </div>}
                  {editingMsgId && <div className="composer-editing-banner">
                    <span>Editing message</span>
                    <button className="btn-dismiss-choice" onClick={cancelEditingMessage} title="Cancel edit"><X size={13} /></button>
                  </div>}
                  <div className="prompt-area">
                    <input ref={attachmentInputRef} type="file" accept="image/*,.glb,.gltf,.obj,.fbx,.ply,.stl" multiple hidden onChange={event => { addAttachments(event.target.files); event.target.value = ''; }} />
                    <button className="btn-attach" onClick={() => attachmentInputRef.current?.click()} title={acceptsImageInput ? 'Attach an image or 3D model' : 'Attach a 3D model (image input needs an image-capable model)'}>
                      <Paperclip size={18} />
                    </button>
                    <button className={`btn-toggle-controls ${autopilot ? 'active' : ''}`} onClick={() => setAutopilot(v => !v)} title="Autopilot: let the model pick among compatible models automatically">
                      <Wand2 size={16} />
                    </button>
                    <button className={`btn-toggle-controls ${reasoningEnabled ? 'active' : ''}`} onClick={() => setReasoningEnabled(v => !v)} title="Reasoning: let the model think before answering">
                      <Brain size={16} />
                    </button>
                    <textarea
                      ref={composerTextareaRef}
                      value={inputPrompt}
                      onChange={e => setInputPrompt(e.target.value)}
                      onKeyDown={e => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                          e.preventDefault();
                          handleComposerSubmit();
                        } else if (e.key === 'Escape' && editingMsgId) {
                          e.preventDefault();
                          cancelEditingMessage();
                        }
                      }}
                      placeholder={acceptsImageInput ? 'Ask about an image or type a message...' : modelStatus.is_loaded ? 'Ask DisposAI anything...' : 'Load a model to start chatting...'}
                    />
                    {isGenerating ? (
                      <button className="btn-send btn-send-stop" onClick={handleStopGeneration} title="Stop generating">
                        <Square size={16} />
                      </button>
                    ) : (
                      <button className="btn-send" onClick={handleComposerSubmit} disabled={isGenerating}>
                        <Send size={18} />
                      </button>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'models' && (
            <div className="tab-panel">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
                <h2 style={{ fontSize: '1.4rem' }}>Models</h2>
                <button className="btn-load-model" onClick={() => { setShowDownloadsPanel(false); setShowModelPicker(true); }}>
                  <PackagePlus size={14} /> Add Model
                </button>
              </div>
              <div className="card" style={{ background: 'none', border: 'none', padding: 0 }}>
                <div className="model-cards-grid" style={{ marginTop: '10px' }}>
                  {modelCards.map(card => {
                    // Backend model_ids include a load timestamp (mdl-{ts}-{port}), so a
                    // model reloaded server-side (e.g. the orchestrator stepping aside for
                    // a run_model call and reloading afterward) gets a fresh id even though
                    // it's the same file. Fall back to matching by the stable modelPath so
                    // this card doesn't show "not loaded" for a model that is actually
                    // loaded under its new id (same pattern as the chat-session resolver).
                    const loadedEntry = loadedModels.find(loaded => loaded.model_id === card.loadedModelId)
                      ?? (card.modelPath ? loadedModels.find(loaded => loaded.model_path === card.modelPath) : null);
                    const isLoaded = Boolean(loadedEntry);
                    return (
                      <div key={card.id} className="model-card-item">
                        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                            <StudioAvatarIcon modelPath={card.modelPath} size={32} />
                            <strong>{card.name}</strong>
                          </div>
                          <button
                            className="btn-remove-card"
                            title="Remove model card"
                            onClick={async () => {
                              if (isLoaded) await handleUnloadModel(loadedEntry.model_id);
                              setModelCards(current => current.filter(item => item.id !== card.id));
                            }}
                          >
                            <X size={13} />
                          </button>
                        </div>
                        <div className="model-card-modality" style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: '0.35rem', color: 'var(--text-secondary)', fontSize: '0.8rem', margin: '0.25rem 0 0.65rem' }}>
                          {card.task_tags?.map(t => <span key={t} className="model-tag-badge">{t.toUpperCase()}</span>)}
                          <span>· {(card.size_bytes / 1e9).toFixed(2)} GB</span>
                        </div>
                        <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', alignItems: 'center', marginTop: '0.25rem' }}>
                          <button className="btn-load-model" onClick={() => openConfigPanel(card)}>
                            <Settings size={14} /> {isLoaded ? 'Settings' : 'Configure'}
                          </button>
                          {isLoaded ? <button className="btn-unload-model" onClick={async () => {
                            await handleUnloadModel(loadedEntry.model_id);
                            setModelCards(current => current.map(item => item.id === card.id ? { ...item, loadedModelId: null } : item));
                          }} disabled={unloadingModelId === loadedEntry.model_id}><Power size={14} /> Eject</button>
                            : <button className="btn-load-model" onClick={() => loadCardModel(card)} disabled={isLoadingModel}><Zap size={14} /> Load</button>}
                          {isLoaded && <button className="btn-load-model" onClick={() => startStudio(loadedEntry, card)} title="Start studio"><Play size={14} /></button>}
                          <button className="btn-load-model" onClick={() => handleOpenModelFolder(card.modelPath)} title="Open containing folder"><Folder size={14} /> Open Folder</button>
                        </div>
                      </div>
                    );
                  })}
                </div>
                {modelCards.length === 0 && <p style={{ color: 'var(--text-secondary)' }}>No model cards yet. Click "Add Model" to add one from the catalog.</p>}
              </div>
            </div>
          )}

          {showModelPicker && (
            <div className="modal-backdrop" onClick={closeModelPicker}>
              <div className="modal-container modal-container-wide" onClick={e => e.stopPropagation()}>
                <div className="modal-header">
                  <h3>Add a model</h3>
                  <button className="config-sidebar-close" onClick={closeModelPicker} title="Close">
                    <X size={16} />
                  </button>
                </div>
                <div className="modal-tabs">
                  <button className={`modal-tab ${modelPickerTab === 'local' ? 'active' : ''}`} onClick={() => setModelPickerTab('local')}>
                    <Folder size={14} /> My Models
                  </button>
                  <button className={`modal-tab ${modelPickerTab === 'discover' ? 'active' : ''}`} onClick={() => setModelPickerTab('discover')}>
                    <Globe size={14} /> Discover
                  </button>
                </div>
                <div className="modal-body">
                  {modelPickerTab === 'local' && (
                    <>
                      <input
                        type="text"
                        className="control-input"
                        placeholder="Search models..."
                        value={modelPickerSearch}
                        onChange={e => setModelPickerSearch(e.target.value)}
                        autoFocus
                      />
                      <div className="modal-list">
                        {detectedModels
                          .filter(model => model.name.toLowerCase().includes(modelPickerSearch.toLowerCase()))
                          .map(model => (
                            <div key={model.path} className="modal-list-item" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '0.5rem' }}>
                              <button
                                className="modal-list-item-main"
                                style={{ flex: 1, textAlign: 'left', background: 'none', border: 'none', padding: 0, cursor: 'pointer', color: 'inherit' }}
                                onClick={() => {
                                  setModelCards(current => [...current, {
                                    id: crypto.randomUUID(),
                                    modelPath: model.path,
                                    name: model.name,
                                    task_tags: model.task_tags,
                                    size_bytes: model.size_bytes,
                                    max_context_size: model.max_context_size ?? null,
                                    image_input_available: model.image_input_available === true,
                                    mmproj_path: model.mmproj_path || null,
                                    mtp_path: model.mtp_path || null,
                                    mesh_vae_path: model.mesh_vae_path || null,
                                    mesh_texgen_path: model.mesh_texgen_path || null,
                                    mesh_components: model.mesh_components || null,
                                    settings: {},
                                  }]);
                                  closeModelPicker();
                                }}
                              >
                                <strong>{model.name}</strong>
                                <span className="modal-list-item-meta">
                                  {model.task_tags?.map(t => t.toUpperCase()).join(', ')}
                                  <span className="hf-param-badge">{(model.size_bytes / 1e9).toFixed(2)} GB</span>
                                </span>
                              </button>
                              <div style={{ display: 'flex', gap: '0.35rem', flexShrink: 0 }}>
                                <button className="btn-load-model" onClick={() => handleOpenModelFolder(model.path)} title="Open containing folder"><Folder size={14} /></button>
                                <button className="btn-remove-card" onClick={() => handleDeleteCatalogModel(model)} title="Delete model from disk"><Trash2 size={13} /></button>
                              </div>
                            </div>
                          ))}
                        {detectedModels.length === 0 && <p style={{ color: 'var(--text-secondary)' }}>No models found in DisposAI's models folder.</p>}
                      </div>
                    </>
                  )}

                  {modelPickerTab === 'discover' && (
                    <div className="hf-discover-layout">
                      <div className="hf-sidebar">
                        <div className="hf-sidebar-top">
                          <span>Filters{hfFilters.length > 0 ? ` (${hfFilters.length})` : ''}</span>
                          {hfFilters.length > 0 && (
                            <button className="hf-clear-filters" onClick={() => setHfFilters([])}>Clear filters</button>
                          )}
                        </div>

                        <div className="hf-sidebar-section">
                          <button className="hf-sidebar-section-header" onClick={() => toggleHfSidebarSection('sort')}>
                            {hfSidebarSections.sort ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
                            Sort By
                          </button>
                          {hfSidebarSections.sort && (
                            <div className="hf-sidebar-section-body">
                              <select className="control-select" value={hfSort} onChange={e => setHfSort(e.target.value)}>
                                <option value="trendingScore">Trending</option>
                                <option value="downloads">Most Downloads</option>
                                <option value="likes">Most Likes</option>
                                <option value="lastModified">Recently Updated</option>
                                <option value="created">Newest</option>
                              </select>
                            </div>
                          )}
                        </div>

                        <div className="hf-sidebar-section">
                          <button className="hf-sidebar-section-header" onClick={() => toggleHfSidebarSection('modelType')}>
                            {hfSidebarSections.modelType ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
                            Model Type
                          </button>
                          {hfSidebarSections.modelType && (
                            <div className="hf-sidebar-section-body hf-checkbox-list">
                              {HF_MODEL_TYPE_GROUPS.map(({ group, types }) => (
                                <React.Fragment key={group}>
                                  <div style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-secondary)', marginTop: '8px', marginBottom: '2px' }}>{group}</div>
                                  {types.map(({ label, tag }) => (
                                    <label key={tag} className="hf-checkbox-item">
                                      <input type="checkbox" checked={hfFilters.includes(tag)} onChange={() => toggleHfFilter(tag)} />
                                      {label}
                                    </label>
                                  ))}
                                </React.Fragment>
                              ))}
                            </div>
                          )}
                        </div>

                        <div className="hf-sidebar-section">
                          <button className="hf-sidebar-section-header" onClick={() => toggleHfSidebarSection('format')}>
                            {hfSidebarSections.format ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
                            Format
                          </button>
                          {hfSidebarSections.format && (
                            <div className="hf-sidebar-section-body hf-checkbox-list">
                              {HF_FORMATS.map(({ label, tag }) => (
                                <label key={tag} className="hf-checkbox-item">
                                  <input type="checkbox" checked={hfFilters.includes(tag)} onChange={() => toggleHfFilter(tag)} />
                                  {label}
                                </label>
                              ))}
                            </div>
                          )}
                        </div>

                        <div className="hf-sidebar-section">
                          <button className="hf-sidebar-section-header" onClick={() => toggleHfSidebarSection('params')}>
                            {hfSidebarSections.params ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
                            Parameters
                          </button>
                          {hfSidebarSections.params && (
                            <div className="hf-sidebar-section-body">
                              <div className="hf-param-range-label">{hfParamRangeLabel()}</div>
                              <div className="hf-param-slider">
                                <div className="hf-param-slider-track" />
                                <div
                                  className="hf-param-slider-fill"
                                  style={{
                                    left: `${(hfParamMinIdx / (HF_PARAM_STOPS.length - 1)) * 100}%`,
                                    right: `${100 - (hfParamMaxIdx / (HF_PARAM_STOPS.length - 1)) * 100}%`,
                                  }}
                                />
                                <input
                                  type="range"
                                  className="hf-param-slider-input"
                                  min={0}
                                  max={HF_PARAM_STOPS.length - 1}
                                  step={1}
                                  value={hfParamMinIdx}
                                  onChange={e => setHfParamMinIdx(Math.min(Number(e.target.value), hfParamMaxIdx))}
                                />
                                <input
                                  type="range"
                                  className="hf-param-slider-input hf-param-slider-input-top"
                                  min={0}
                                  max={HF_PARAM_STOPS.length - 1}
                                  step={1}
                                  value={hfParamMaxIdx}
                                  onChange={e => setHfParamMaxIdx(Math.max(Number(e.target.value), hfParamMinIdx))}
                                />
                              </div>
                            </div>
                          )}
                        </div>

                        <div className="hf-sidebar-section">
                          <button className="hf-sidebar-section-header" onClick={() => toggleHfSidebarSection('language')}>
                            {hfSidebarSections.language ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
                            Language
                          </button>
                          {hfSidebarSections.language && (
                            <div className="hf-sidebar-section-body hf-checkbox-list">
                              {HF_LANGUAGES.map(({ label, tag }) => (
                                <label key={tag} className="hf-checkbox-item">
                                  <input type="checkbox" checked={hfFilters.includes(tag)} onChange={() => toggleHfFilter(tag)} />
                                  {label}
                                </label>
                              ))}
                            </div>
                          )}
                        </div>

                        <div className="hf-sidebar-section">
                          <button className="hf-sidebar-section-header" onClick={() => toggleHfSidebarSection('precision')}>
                            {hfSidebarSections.precision ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
                            Precision / Quantization
                          </button>
                          {hfSidebarSections.precision && (
                            <div className="hf-sidebar-section-body hf-checkbox-list">
                              {HF_PRECISIONS.map(({ label, tag }) => (
                                <label key={tag} className="hf-checkbox-item">
                                  <input type="checkbox" checked={hfFilters.includes(tag)} onChange={() => toggleHfFilter(tag)} />
                                  {label}
                                </label>
                              ))}
                            </div>
                          )}
                        </div>
                      </div>

                      <div className="hf-results-panel">
                        <div className="hf-search-bar">
                          <input
                            type="text"
                            className="control-input"
                            placeholder="Search Hugging Face for models..."
                            value={hfSearchQuery}
                            onChange={e => setHfSearchQuery(e.target.value)}
                            onKeyDown={e => { if (e.key === 'Enter') runHfSearch(hfSearchQuery, hfSort, hfFilters, ...hfParamBounds()); }}
                            autoFocus
                          />
                        </div>

                        <div className="modal-list">
                          {hfSearchLoading && (
                            <div className="hf-empty-state">
                              <Loader size={22} className="spin" />
                              <p>Searching Hugging Face...</p>
                            </div>
                          )}
                          {!hfSearchLoading && hfSearchError && (
                            <div className="hf-empty-state">
                              <XCircle size={22} />
                              <p>Search failed. Check your internet connection.</p>
                            </div>
                          )}
                          {!hfSearchLoading && !hfSearchError && hfSearchResults.length === 0 && (
                            <div className="hf-empty-state">
                              <Search size={22} />
                              <p>{hfSearchQuery.trim() ? 'No results found.' : 'No models match these filters.'}</p>
                            </div>
                          )}
                          {!hfSearchLoading && !hfSearchError && hfSearchResults.map(result => {
                            const repoId = result.id || `${result.author}/${result.modelId}`;
                            const isSelected = hfSelectedRepo === repoId;
                            return (
                              <div
                                key={repoId}
                                className={`hf-result-card${isSelected ? ' hf-result-card-selected' : ''}`}
                              >
                                <div className="hf-result-header" onClick={() => selectHfRepo(repoId)}>
                                  <div>
                                    <strong>{result.modelId || repoId}</strong>
                                    <span className="modal-list-item-meta">
                                      {' · '}
                                      {result.author && (
                                        <img
                                          className="hf-author-avatar"
                                          src={`http://127.0.0.1:8080/v1/model/hf-avatar?author=${encodeURIComponent(result.author)}&v=2`}
                                          alt=""
                                          onError={(e) => { e.currentTarget.style.display = 'none'; }}
                                        />
                                      )}
                                      {result.author}
                                    </span>
                                    {formatParamCount(result.params) && (
                                      <span className="hf-param-badge">{formatParamCount(result.params)}</span>
                                    )}
                                  </div>
                                  <div className="hf-result-stats">
                                    <span><Download size={12} /> {formatCount(result.downloads)}</span>
                                    <span><Heart size={12} /> {formatCount(result.likes)}</span>
                                  </div>
                                </div>
                                {result.tags?.length > 0 && (
                                  <div className="hf-result-tags">
                                    {result.tags.slice(0, 6).map(tag => (
                                      <span key={tag} className="hf-tag-badge">{tag}</span>
                                    ))}
                                  </div>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      </div>

                      <div className="hf-detail-panel">
                        {!hfSelectedRepo && (
                          <div className="hf-detail-placeholder">
                            <Box size={22} />
                            <p>Select a model to see details</p>
                          </div>
                        )}
                        {hfSelectedRepo && (() => {
                          const selected = hfSearchResults.find(r => (r.id || `${r.author}/${r.modelId}`) === hfSelectedRepo);
                          return (
                            <>
                              <div className="hf-detail-header">
                                <div className="hf-detail-title">
                                  <strong>{selected?.modelId || hfSelectedRepo}</strong>
                                  <button
                                    className="hf-detail-link"
                                    title="Open on Hugging Face"
                                    onClick={() => window.require('electron').shell.openExternal(`https://huggingface.co/${hfSelectedRepo}`)}
                                  >
                                    <ArrowUpRight size={14} />
                                  </button>
                                  {formatParamCount(selected?.params) && (
                                    <span className="hf-param-badge">{formatParamCount(selected.params)}</span>
                                  )}
                                  {selected?.author && <div className="modal-list-item-meta">{selected.author}</div>}
                                </div>
                                <button className="hf-detail-close" onClick={() => selectHfRepo(hfSelectedRepo)} title="Close">
                                  <X size={16} />
                                </button>
                              </div>

                              <div className="hf-detail-stats">
                                <span><Download size={13} /> {formatCount(selected?.downloads)} downloads</span>
                                <span><Heart size={13} /> {formatCount(selected?.likes)} likes</span>
                              </div>

                              {selected?.tags?.length > 0 && (
                                <div className="hf-detail-section">
                                  <div className="hf-detail-section-title">About</div>
                                  <div className="hf-result-tags">
                                    {selected.tags.map(tag => (
                                      <span key={tag} className="hf-tag-badge">{tag}</span>
                                    ))}
                                  </div>
                                </div>
                              )}

                              <div className="hf-detail-section">
                                <div className="hf-detail-section-title">Versions</div>
                                {hfRepoFilesLoading && (
                                  <div className="hf-empty-state">
                                    <Loader size={18} className="spin" />
                                  </div>
                                )}
                                {!hfRepoFilesLoading && hfRepoFiles.length === 0 && (
                                  <p className="modal-list-item-meta">No downloadable files found in this repo.</p>
                                )}
                                {!hfRepoFilesLoading && hfRepoFiles.length > 0 && (
                                  <div className="hf-files-list">
                                    <input
                                      type="text"
                                      className="control-input"
                                      placeholder="Filter files..."
                                      value={hfFileSearch}
                                      onChange={e => setHfFileSearch(e.target.value)}
                                    />
                                    <div className="hf-file-row hf-file-row-all">
                                      <div className="modal-list-item-meta">Download every file into a per-model folder</div>
                                      <div className="hf-file-actions">
                                        {['sharded', 'variants', 'components'].includes(hfRepoKind) && hfAutodownload.length > 0 && (
                                          <button className="hf-download-btn hf-autodownload-btn" onClick={() => startHfAutodownload(hfSelectedRepo)}>
                                            <Download size={14} /> Autodownload
                                          </button>
                                        )}
                                        <button className="hf-download-btn" onClick={() => startHfDownloadAll(hfSelectedRepo)}>
                                          <Download size={14} /> Download all files
                                        </button>
                                      </div>
                                    </div>
                                    {hfAutodownloadReason && (
                                      <div className="hf-autodownload-reason">{hfAutodownloadReason}</div>
                                    )}
                                    {(() => {
                                      const groups = new Map();
                                      const visibleFiles = hfFileSearch.trim()
                                        ? hfRepoFiles.filter(file => file.filename.toLowerCase().includes(hfFileSearch.trim().toLowerCase()))
                                        : hfRepoFiles;
                                      visibleFiles.forEach(file => {
                                        const slashIdx = file.filename.lastIndexOf('/');
                                        const dir = slashIdx === -1 ? '' : file.filename.slice(0, slashIdx);
                                        if (!groups.has(dir)) groups.set(dir, []);
                                        groups.get(dir).push(file);
                                      });
                                      return Array.from(groups.entries()).map(([dir, files]) => {
                                        const depth = dir === '' ? 0 : dir.split('/').length;
                                        const collapsed = dir !== '' && hfCollapsedFolders.has(dir);
                                        return (
                                          <div key={dir || '__root__'} className="hf-folder-group">
                                            {dir !== '' && (
                                              <div
                                                className="hf-folder-header"
                                                style={{ paddingLeft: `${(depth - 1) * 1.1}rem` }}
                                                onClick={() => setHfCollapsedFolders(current => {
                                                  const next = new Set(current);
                                                  if (next.has(dir)) next.delete(dir); else next.add(dir);
                                                  return next;
                                                })}
                                              >
                                                <ChevronRight size={12} className={`hf-folder-chevron${collapsed ? '' : ' hf-folder-chevron-open'}`} />
                                                {dir}/ <span className="hf-folder-count">({files.length})</span>
                                              </div>
                                            )}
                                            {!collapsed && files.map(file => {
                                              const download = hfDownloads[`${hfSelectedRepo}::${file.filename}`];
                                              const progressPct = download && download.total_bytes > 0
                                                ? Math.min(100, (download.downloaded_bytes / download.total_bytes) * 100)
                                                : 0;
                                              const roleInfo = getHfFileRoleInfo(file);
                                              const baseName = dir === '' ? file.filename : file.filename.slice(dir.length + 1);
                                              return (
                                                <div key={file.filename} className="hf-file-row" data-filename={file.filename} style={{ paddingLeft: `${depth * 1.1}rem` }}>
                                                  <div className="hf-file-info">
                                                    <div className="hf-file-name-row">
                                                      <span className="hf-file-role" title={roleInfo.tooltip}>{roleInfo.icon}</span>
                                                      <span className="hf-file-name">{baseName}</span>
                                                      {file.size != null && <span className="hf-file-size">{formatFileSize(file.size)}</span>}
                                                      {file.quant && <span className="hf-file-quant">{file.quant}</span>}
                                                    </div>
                                                    {download && download.status === 'downloading' && (
                                                      <>
                                                        <div className="hf-progress-bar">
                                                          <div className="hf-progress-fill" style={{ width: `${progressPct}%` }} />
                                                        </div>
                                                        <div className="hf-progress-row">
                                                          <span className="hf-progress-text">
                                                            {formatFileSize(download.downloaded_bytes)} / {formatFileSize(download.total_bytes)}
                                                          </span>
                                                          <button
                                                            className="hf-cancel-btn"
                                                            onClick={() => cancelHfDownload(hfSelectedRepo, file.filename)}
                                                            title="Cancel download"
                                                          >
                                                            <X size={12} />
                                                          </button>
                                                        </div>
                                                      </>
                                                    )}
                                                    {download && download.status === 'error' && (
                                                      <span className="modal-list-item-meta" style={{ color: '#ef4444' }}>
                                                        Download failed{download.error ? `: ${download.error}` : ''}
                                                        {download.error && (download.error.includes('403') || download.error.toLowerCase().includes('forbidden')) && (
                                                          <span style={{ display: 'block', marginTop: 2, fontSize: 11 }}>
                                                            This model requires license acceptance —{' '}
                                                            <a href={`https://huggingface.co/${hfSelectedRepo}`} target="_blank" rel="noreferrer" style={{ color: '#60a5fa', textDecoration: 'underline' }}>open model page</a>
                                                            {' '}to accept the terms, then retry.
                                                          </span>
                                                        )}
                                                        {download.error && (download.error.includes('401') || download.error.toLowerCase().includes('unauthorized')) && (
                                                          <span style={{ display: 'block', marginTop: 2, fontSize: 11 }}>
                                                            Authentication required — add your HF token in{' '}
                                                            <a href="#" onClick={e => { e.preventDefault(); setActiveTab('settings'); }} style={{ color: '#60a5fa', textDecoration: 'underline' }}>General Settings</a>
                                                          </span>
                                                        )}
                                                      </span>
                                                    )}
                                                  </div>
                                                  <div className="hf-file-actions">
                                                    {download && download.status === 'complete' ? (
                                                      <span className="hf-download-btn hf-download-done"><Check size={14} /> Done</span>
                                                    ) : (
                                                      <button
                                                        className="hf-download-btn"
                                                        disabled={download?.status === 'downloading'}
                                                        onClick={() => startHfDownload(hfSelectedRepo, file.filename)}
                                                      >
                                                        {download?.status === 'downloading' ? <Loader size={14} className="spin" /> : <Download size={14} />}
                                                        {download?.status === 'downloading' ? 'Downloading' : 'Download'}
                                                      </button>
                                                    )}
                                                  </div>
                                                </div>
                                              );
                                            })}
                                          </div>
                                        );
                                      });
                                    })()}
                                  </div>
                                )}
                              </div>
                            </>
                          );
                        })()}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {activeTab === 'settings' && (
            <div className="tab-panel">
              <h2 style={{ fontSize: '1.4rem', marginBottom: '0.3rem' }}>General Settings</h2>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '1.5rem' }}>
                Preferences for this local Dispos Studio app. Changes are saved on this device.
              </p>
              <div className="card" style={{ maxWidth: '680px' }}>
                <div className="slider-header" style={{ padding: '0.8rem 0', borderBottom: '1px solid var(--border-color)' }}>
                  <div><strong>Auto-select newly loaded models</strong><div className="slider-hint">Use the latest loaded model for chat automatically.</div></div>
                  <input type="checkbox" checked={appSettings.autoSelectNewest} onChange={e => setAppSettings(current => ({ ...current, autoSelectNewest: e.target.checked }))} />
                </div>
                <div style={{ padding: '1rem 0', borderBottom: '1px solid var(--border-color)' }}>
                  <div className="slider-header"><div><strong>Model status refresh</strong><div className="slider-hint">How often Dispos Studio checks loaded models.</div></div><span className="badge-value">{appSettings.refreshSeconds}s</span></div>
                  <input className="control-slider" type="range" min="1" max="10" step="1" value={appSettings.refreshSeconds} onChange={e => setAppSettings(current => ({ ...current, refreshSeconds: Number(e.target.value) }))} />
                </div>
                <div className="slider-header" style={{ padding: '1rem 0', borderBottom: '1px solid var(--border-color)' }}>
                  <div><strong>Show thought process panels</strong><div className="slider-hint">Display model reasoning when it is supplied in a response.</div></div>
                  <input type="checkbox" checked={appSettings.showThoughtProcess} onChange={e => setAppSettings(current => ({ ...current, showThoughtProcess: e.target.checked }))} />
                </div>
                <div className="slider-header" style={{ padding: '1rem 0', borderBottom: '1px solid var(--border-color)' }}>
                  <div><strong>Autopilot (auto model-selection)</strong><div className="slider-hint">When on, the model picks among compatible models automatically instead of asking you.</div></div>
                  <input type="checkbox" checked={appSettings.autopilot} onChange={e => setAppSettings(current => ({ ...current, autopilot: e.target.checked }))} />
                </div>
                <div className="slider-header" style={{ paddingTop: '1rem' }}>
                  <div><strong>Generated media retention</strong><div className="slider-hint">How long generated images/audio/meshes are kept before being cleaned up. "Forever" saves generated assets to disk permanently.</div></div>
                  <select
                    className="control-select"
                    value={`${appSettings.mediaRetention.ttl_seconds}:${appSettings.mediaRetention.persist_disk}`}
                    onChange={e => {
                      const [ttl, persist] = e.target.value.split(':');
                      const mediaRetention = { ttl_seconds: Number(ttl), persist_disk: persist === 'true' };
                      setAppSettings(current => ({ ...current, mediaRetention }));
                      fetch('http://127.0.0.1:8080/v1/config/media-retention', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(mediaRetention),
                      }).catch(() => { });
                    }}
                  >
                    <option value="1800:false">30 minutes</option>
                    <option value="3600:false">1 hour</option>
                    <option value="21600:false">6 hours</option>
                    <option value="86400:false">24 hours</option>
                    <option value="0:true">Forever (save to disk)</option>
                  </select>
                </div>
                <div style={{ padding: '1rem 0', borderTop: '1px solid var(--border-color)' }}>
                  <div className="slider-header"><div><strong>HuggingFace token</strong><div className="slider-hint">Required for downloading gated models. Get yours at huggingface.co/settings/tokens</div></div>{hfHasToken && <span style={{ color: 'var(--accent-green)', fontSize: '0.8rem', fontWeight: 600 }}>Active</span>}</div>
                  <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
                    <input
                      type="password"
                      className="control-input"
                      placeholder="hf_..."
                      value={hfTokenInput}
                      onChange={e => setHfTokenInput(e.target.value)}
                      onKeyDown={e => e.key === 'Enter' && saveHfToken()}
                      style={{ flex: 1 }}
                    />
                    <button className="btn-primary" onClick={saveHfToken} disabled={!hfTokenInput.trim()} style={{ padding: '0.4rem 1rem' }}>
                      Save
                    </button>
                  </div>
                  {hfTokenSaved && <div style={{ color: 'var(--accent-green)', fontSize: '0.8rem', marginTop: 4 }}>Token saved!</div>}
                </div>
              </div>
            </div>
          )}

          {/* Embeddings */}
          {activeTab === 'embeddings' && (
            <div className="tab-panel">
              <div className="model-status-strip">
                <div className="status-info">
                  <StatusDot state={activeEmbedDotState} />
                  <div className="status-text">
                    <span className="model-name">
                      {activeEmbedLoaded
                        ? activeEmbedLoaded.model_path?.split('\\').pop() || 'Model Loaded'
                        : embedModelId || 'No Model Loaded'}
                    </span>
                    <span className="model-meta">
                      {activeEmbedLoaded ? 'Ready to generate embeddings' : embedModelId ? 'Not currently loaded — load it to use this studio' : 'Load a text model to generate embeddings'}
                    </span>
                  </div>
                </div>
              </div>
              <h2 style={{ fontSize: '1.4rem', marginBottom: '0.3rem' }}>Embeddings</h2>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '1.5rem' }}>
                Generate embedding vectors from a loaded text model.
              </p>
              <div className="card" style={{ maxWidth: '680px' }}>
                <div className="form-group">
                  <label>Model</label>
                  <select
                    className="control-select"
                    value={embedModelId}
                    onChange={e => setEmbedModelId(e.target.value)}
                  >
                    <option value="">First loaded text model</option>
                    {loadedModels.filter(model => categoryForTags(model.task_tags) === 'chat').map(model => (
                      <option key={model.model_id} value={model.model_id}>
                        {model.model_path?.split('\\').pop() ?? model.model_id}
                      </option>
                    ))}
                  </select>
                </div>
                <div className="form-group">
                  <label>Input Text <span style={{ fontWeight: 400, color: 'var(--text-secondary)' }}>(one entry per line)</span></label>
                  <textarea
                    className="control-input"
                    rows={5}
                    value={embedInput}
                    onChange={e => setEmbedInput(e.target.value)}
                  />
                </div>
                <div className="gen-btn-row">
                  <button className="btn-primary" onClick={handleGenerateEmbeddings} disabled={isEmbedding || !embedInput.trim()}>
                    <Search size={16} /> {isEmbedding ? 'Embedding...' : 'Generate Embeddings'}
                  </button>
                </div>
                {embedResults && (
                  <div style={{ marginTop: '1rem' }}>
                    <div className="slider-hint" style={{ marginBottom: '0.5rem' }}>
                      {embedResults.data?.length ?? 0} vector(s) · model: {embedResults.model}
                    </div>
                    {(embedResults.data || []).map(item => (
                      <div key={item.index} style={{ padding: '0.5rem 0', borderTop: '1px solid var(--border-color)', fontSize: '0.8rem', fontFamily: 'monospace' }}>
                        <div>[{item.index}] dim={item.embedding.length}</div>
                        <div style={{ color: 'var(--text-secondary)', wordBreak: 'break-all' }}>
                          [{item.embedding.slice(0, 8).map(v => v.toFixed(4)).join(', ')}{item.embedding.length > 8 ? ', ...' : ''}]
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* 2. Image Studio */}
          {activeTab === 'image' && (
            <div className="tab-panel">
              <div className="model-status-strip">
                <div className="status-info">
                  <StatusDot state={modelDotState(activeImageStudio, activeImageLoaded)} />
                  <div className="status-text">
                    <span className="model-name">
                      {activeImageLoaded
                        ? activeImageLoaded.model_path?.split('\\').pop() || activeImageStudio?.name || 'Model Loaded'
                        : activeImageStudio?.name || 'No Model Loaded'}
                    </span>
                    <span className="model-meta">
                      {activeImageLoaded ? 'Ready for image generation' : activeImageStudio ? 'Not currently loaded — load it to use this studio' : 'Use the sidebar to configure and load a model'}
                    </span>
                  </div>
                </div>
              </div>

              <h2 style={{ fontSize: '1.4rem', marginBottom: '0.3rem' }}>Stable Diffusion Image Studio</h2>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '1.5rem' }}>
                {imageStudioSupportsImg2Img
                  ? 'Local text-to-image and image-to-image synthesis using stable-diffusion.cpp'
                  : 'Local text-to-image synthesis using stable-diffusion.cpp'}
              </p>

              <div className="grid-2">
                <div className="card">
                  {(!activeImageStudio || imageStudioSupportsImg2Img) && (
                    <div className="form-group">
                      <label>Reference Image <span style={{ fontWeight: 400, color: 'var(--text-secondary)' }}>(optional)</span></label>
                      <input
                        ref={imgInitImageInputRef}
                        type="file"
                        accept="image/*"
                        style={{ display: 'none' }}
                        onChange={e => { handleImgInitImage(e.target.files); e.target.value = ''; }}
                      />
                      {imgInitImage ? (
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                          <div className="mesh3d-dropzone-thumb" style={{ width: 80, height: 80 }}>
                            <img src={imgInitImage} alt={imgInitImageName} />
                            <button type="button" className="mesh3d-thumb-remove" onClick={() => { setImgInitImage(null); setImgInitImageName(null); }} title="Remove"><X size={12} /></button>
                          </div>
                          <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', wordBreak: 'break-all' }}>{imgInitImageName}</span>
                        </div>
                      ) : (
                        <div className="mesh3d-dropzone" onClick={() => imgInitImageInputRef.current?.click()}>
                          <Paperclip size={20} />
                          <span>Click to upload an image</span>
                        </div>
                      )}
                      {imgInitImage && (
                        <div style={{ marginTop: '0.75rem' }}>
                          <label>Strength ({Number(imgStrength).toFixed(2)})</label>
                          <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.05"
                            value={imgStrength}
                            onChange={e => setImgStrength(e.target.value)}
                          />
                        </div>
                      )}
                    </div>
                  )}

                  <div className="form-group">
                    <label>Prompt</label>
                    <input
                      type="text"
                      value={imgPrompt}
                      onChange={e => setImgPrompt(e.target.value)}
                    />
                  </div>

                  <div className="form-group">
                    <label>Negative Prompt</label>
                    <input
                      type="text"
                      value={imgNegativePrompt}
                      onChange={e => setImgNegativePrompt(e.target.value)}
                    />
                  </div>

                  <div style={{ display: 'flex', gap: '1rem' }}>
                    <div className="form-group" style={{ flex: 1 }}>
                      <label>Steps</label>
                      <input
                        type="number"
                        value={imgSteps}
                        onChange={e => setImgSteps(e.target.value)}
                      />
                    </div>
                    <div className="form-group" style={{ flex: 1 }}>
                      <label>CFG Scale</label>
                      <input
                        type="number"
                        step="0.5"
                        value={imgCfgScale}
                        onChange={e => setImgCfgScale(e.target.value)}
                      />
                    </div>
                  </div>

                  <div style={{ display: 'flex', gap: '1rem' }}>
                    <div className="form-group" style={{ flex: 1 }}>
                      <label>Width</label>
                      <input
                        type="number"
                        step="64"
                        value={imgWidth}
                        onChange={e => setImgWidth(e.target.value)}
                      />
                    </div>
                    <div className="form-group" style={{ flex: 1 }}>
                      <label>Height</label>
                      <input
                        type="number"
                        step="64"
                        value={imgHeight}
                        onChange={e => setImgHeight(e.target.value)}
                      />
                    </div>
                  </div>

                  <div className="form-group">
                    <label>Seed <span style={{ fontWeight: 400, color: 'var(--text-secondary)' }}>(-1 = random)</span></label>
                    <input
                      type="number"
                      value={imgSeed}
                      onChange={e => setImgSeed(e.target.value)}
                    />
                  </div>

                  <div className="gen-btn-row">
                    <button className="btn-primary" onClick={handleGenerateImage} disabled={isGeneratingImg}>
                      <Sparkles size={16} /> {isGeneratingImg ? 'Rendering...' : 'Generate Image'}
                    </button>
                    {isGeneratingImg && (
                      <button className="btn-cancel-gen" onClick={() => cancelGenerationJob(activeImageJobId)}>
                        <X size={16} /> Cancel
                      </button>
                    )}
                  </div>
                  {isGeneratingImg && <GenProgressBar progress={imgProgress} />}
                  {installProgress && <GenProgressBar progress={installProgress} />}

                  {imgMissingComponents && imgMissingComponents.length > 0 && (
                    <div className="hf-folder-group" style={{ marginTop: '1rem', border: '1px solid var(--border-color, #333)', borderRadius: 6, padding: '0.75rem' }}>
                      <div className="hf-folder-header" style={{ paddingLeft: 0 }}>
                        Missing model components
                      </div>
                      {imgMissingComponents.map((comp, idx) => {
                        const download = comp.source ? hfDownloads[`${comp.source.repo}::${comp.source.filename}`] : null;
                        const progressPct = download && download.total_bytes > 0
                          ? Math.min(100, (download.downloaded_bytes / download.total_bytes) * 100)
                          : 0;
                        return (
                          <div key={idx} className="hf-file-row" style={{ paddingLeft: 0 }}>
                            <div className="hf-file-info">
                              <div className="hf-file-name-row">
                                <span className="hf-file-name">{comp.kind_name}</span>
                              </div>
                              {comp.source ? (
                                <>
                                  {(!download || download.status === 'error') && (
                                    <button
                                      className="hf-download-btn"
                                      onClick={() => startHfDownload(comp.source.repo, comp.source.filename, comp.target_path, comp.source.target_filename)}
                                    >
                                      Download {comp.source.filename} ({comp.source.repo})
                                    </button>
                                  )}
                                  {download && download.status === 'downloading' && (
                                    <>
                                      <div className="hf-progress-bar">
                                        <div className="hf-progress-fill" style={{ width: `${progressPct}%` }} />
                                      </div>
                                      <div className="hf-progress-row">
                                        <span className="hf-progress-text">
                                          {formatFileSize(download.downloaded_bytes)} / {formatFileSize(download.total_bytes)}
                                        </span>
                                      </div>
                                    </>
                                  )}
                                  {download && download.status === 'complete' && (
                                    <span className="modal-list-item-meta">Downloaded — retrying generation…</span>
                                  )}
                                  {download && download.status === 'error' && (
                                    <span className="modal-list-item-meta" style={{ color: '#ef4444' }}>
                                      Download failed{download.error ? `: ${download.error}` : ''}
                                    </span>
                                  )}
                                </>
                              ) : (
                                <span className="modal-list-item-meta">
                                  No known auto-download source — place a file manually at: {comp.target_path}
                                </span>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>

                <div className="card">
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <label>Generated Canvas Output</label>
                    {imgSrc && (
                      <button type="button" className="image-download-btn" onClick={handleDownloadImage} title="Download image">
                        <Download size={14} /> Download
                      </button>
                    )}
                  </div>
                  <div className="preview-box">
                    {imgSrc ? (
                      <img src={imgSrc} alt="SD Output" />
                    ) : (
                      <>
                        <Image size={40} />
                        <span>Rendered output image will display here</span>
                      </>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* 3. Voice Studio */}
          {activeTab === 'audio' && (
            <div className="tab-panel">
              <div className="model-status-strip">
                <div className="status-info">
                  <StatusDot state={modelDotState(activeTtsStudio, activeTtsLoaded)} />
                  <div className="status-text">
                    <span className="model-name">
                      {activeTtsLoaded
                        ? activeTtsLoaded.model_path?.split('\\').pop() || activeTtsStudio?.name || 'Model Loaded'
                        : activeTtsStudio?.name || 'No Model Loaded'}
                    </span>
                    <span className="model-meta">
                      {activeTtsLoaded ? 'Ready to synthesize speech' : activeTtsStudio ? 'Not currently loaded — load it to use this studio' : 'Use the sidebar to configure and load a model'}
                    </span>
                  </div>
                </div>
              </div>

              <h2 style={{ fontSize: '1.4rem', marginBottom: '0.3rem' }}>Voice Studio</h2>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '1.5rem' }}>
                Kokoro Text-to-Speech
              </p>

              <div className="grid-2">
                <div className="card">
                  <h3>Text-to-Speech Synthesizer (Kokoro)</h3>
                  <br />
                  <div className="form-group">
                    <label>Input Text</label>
                    <input
                      type="text"
                      value={ttsInput}
                      onChange={e => setTtsInput(e.target.value)}
                    />
                  </div>

                  <div className="form-group">
                    <label>Speed ({Number(ttsSpeed).toFixed(2)}x)</label>
                    <input
                      type="range"
                      min="0.5"
                      max="2"
                      step="0.05"
                      value={ttsSpeed}
                      onChange={e => setTtsSpeed(Number(e.target.value))}
                    />
                  </div>

                  <div className="gen-btn-row">
                    <button className="btn-primary" onClick={handleSynthesizeSpeech} disabled={isGeneratingTts}>
                      <Play size={16} /> {isGeneratingTts ? 'Synthesizing...' : 'Synthesize Speech'}
                    </button>
                    {isGeneratingTts && (
                      <button className="btn-cancel-gen" onClick={() => cancelGenerationJob(activeTtsJobId)}>
                        <X size={16} /> Cancel
                      </button>
                    )}
                  </div>
                  {isGeneratingTts && <GenProgressBar progress={ttsProgress} />}
                  {installProgress && <GenProgressBar progress={installProgress} />}

                  {audioSrc && <audio controls autoPlay src={audioSrc} style={{ width: '100%', marginTop: '1rem' }} />}
                </div>
              </div>
            </div>
          )}

          {/* Transcribe Studio */}
          {activeTab === 'transcribe' && (
            <div className="tab-panel">
              <div className="model-status-strip">
                <div className="status-info">
                  <StatusDot state={modelDotState(activeAsrStudio, activeAsrLoaded)} />
                  <div className="status-text">
                    <span className="model-name">
                      {activeAsrLoaded
                        ? activeAsrLoaded.model_path?.split('\\').pop() || activeAsrStudio?.name || 'Model Loaded'
                        : activeAsrStudio?.name || 'No Model Loaded'}
                    </span>
                    <span className="model-meta">
                      {activeAsrLoaded ? 'Ready to transcribe audio' : activeAsrStudio ? 'Not currently loaded — load it to use this studio' : 'Use the sidebar to configure and load a model'}
                    </span>
                  </div>
                </div>
              </div>

              <h2 style={{ fontSize: '1.4rem', marginBottom: '0.3rem' }}>Transcribe Studio</h2>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '1.5rem' }}>
                Whisper Speech-to-Text ASR
              </p>

              <div className="grid-2">
                <div className="card">
                  <h3>Speech-to-Text ASR (Whisper)</h3>
                  <br />
                  <input
                    ref={asrFileInputRef}
                    type="file"
                    accept="audio/*"
                    style={{ display: 'none' }}
                    onChange={e => { handleTranscribeAudio(e.target.files); e.target.value = ''; }}
                  />
                  <div className="preview-box" style={{ height: '180px' }} onClick={() => asrFileInputRef.current?.click()}>
                    <FileAudio size={40} />
                    <span>{asrFileName || 'Drop audio file for transcription'}</span>
                  </div>

                  <div className="form-group" style={{ marginTop: '1rem' }}>
                    <label>Language <span style={{ fontWeight: 400, color: 'var(--text-secondary)' }}>(optional)</span></label>
                    <input
                      type="text"
                      placeholder="e.g. en"
                      value={asrLanguage}
                      onChange={e => setAsrLanguage(e.target.value)}
                    />
                  </div>
                </div>

                <div className="card">
                  <label>Transcription Result</label>
                  <textarea
                    readOnly
                    value={isTranscribing ? 'Transcribing...' : asrText}
                    placeholder="Transcribed text will appear here."
                    style={{ width: '100%', minHeight: '180px', marginTop: '0.75rem', resize: 'vertical' }}
                  />
                </div>
              </div>
            </div>
          )}

          {/* 4. Video Studio */}
          {activeTab === 'video' && (
            <div className="tab-panel">
              <div className="model-status-strip">
                <div className="status-info">
                  <StatusDot state={modelDotState(activeVideoStudio, activeVideoLoaded)} />
                  <div className="status-text">
                    <span className="model-name">
                      {activeVideoLoaded
                        ? activeVideoLoaded.model_path?.split('\\').pop() || activeVideoStudio?.name || 'Model Loaded'
                        : activeVideoStudio?.name || 'No Model Loaded'}
                    </span>
                    <span className="model-meta">
                      {activeVideoLoaded ? 'Ready for video generation' : activeVideoStudio ? 'Not currently loaded — load it to use this studio' : 'Use the sidebar to configure and load a model'}
                    </span>
                  </div>
                </div>
              </div>

              <h2 style={{ fontSize: '1.4rem', marginBottom: '0.3rem' }}>Wan Video Runner Studio</h2>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '1.5rem' }}>
                Text-to-Video generation engine
              </p>

              <div className="grid-2">
                <div className="card">
                  {videoSchemaSupportsImage && (
                    <div className="form-group">
                      <label>Conditioning Image <span style={{ fontWeight: 400, color: 'var(--text-secondary)' }}>(optional, image-to-video)</span></label>
                      <input
                        ref={videoInitImageInputRef}
                        type="file"
                        accept="image/*"
                        style={{ display: 'none' }}
                        onChange={e => { handleVideoInitImage(e.target.files); e.target.value = ''; }}
                      />
                      {videoInitImage ? (
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                          <div className="mesh3d-dropzone-thumb" style={{ width: 80, height: 80 }}>
                            <img src={videoInitImage} alt={videoInitImageName} />
                            <button type="button" className="mesh3d-thumb-remove" onClick={() => { setVideoInitImage(null); setVideoInitImageName(null); }} title="Remove"><X size={12} /></button>
                          </div>
                          <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', wordBreak: 'break-all' }}>{videoInitImageName}</span>
                        </div>
                      ) : (
                        <div className="mesh3d-dropzone" onClick={() => videoInitImageInputRef.current?.click()}>
                          <Paperclip size={20} />
                          <span>Click to upload an image</span>
                        </div>
                      )}
                    </div>
                  )}

                  <div className="form-group">
                    <label>Video Prompt</label>
                    <input
                      type="text"
                      value={videoPrompt}
                      onChange={e => setVideoPrompt(e.target.value)}
                    />
                  </div>

                  {/* Every field but the prompt above is rendered generically
                      from GET /v1/videos/schema, which reflects what the
                      loaded pipeline's __call__ signature actually accepts
                      (e.g. height/width/negative_prompt only show up when
                      the loaded architecture supports them). */}
                  {videoSchemaParams.map(p => (
                    <SchemaParamField
                      key={p.name}
                      param={p}
                      value={videoParamValues[p.name]}
                      onChange={val => setVideoParam(p.name, val)}
                    />
                  ))}
                  {!videoSchema && (
                    <p className="slider-hint" style={{ marginBottom: '1rem' }}>
                      Generation parameters are unavailable (could not reach the daemon's schema endpoint — load a video model first).
                    </p>
                  )}

                  <div className="gen-btn-row">
                    <button className="btn-primary" onClick={handleGenerateVideo} disabled={isGeneratingVideo}>
                      <Video size={16} /> {isGeneratingVideo ? 'Rendering...' : 'Render Video'}
                    </button>
                    {isGeneratingVideo && (
                      <button className="btn-cancel-gen" onClick={() => cancelGenerationJob(activeVideoJobId)}>
                        <X size={16} /> Cancel
                      </button>
                    )}
                  </div>
                  {isGeneratingVideo && <GenProgressBar progress={videoProgress} />}
                  {modelLoadProgress && <GenProgressBar progress={modelLoadProgress} />}
                  {installProgress && <GenProgressBar progress={installProgress} />}
                </div>

                <div className="card">
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <label>Generated Video Output</label>
                    {videoSrc && (
                      <button type="button" className="image-download-btn" onClick={() => handleDownloadMedia(videoSrc, 'video/mp4')} title="Download video">
                        <Download size={14} /> Download
                      </button>
                    )}
                  </div>
                  <div className="preview-box" style={{ marginTop: '1rem' }}>
                    {videoSrc ? (
                      <VideoPlayer src={videoSrc} autoPlay style={{ width: '100%' }} />
                    ) : (
                      <>
                        <Play size={45} />
                        <span>Video player output preview</span>
                      </>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* 5. 3D Model Studio */}
          {activeTab === 'mesh3d' && (
            <div className="tab-panel">
              <div className="model-status-strip">
                <div className="status-info">
                  <StatusDot state={modelDotState(activeMesh3dStudio, activeMesh3dLoaded)} />
                  <div className="status-text">
                    <span className="model-name">
                      {activeMesh3dLoaded
                        ? activeMesh3dLoaded.model_path?.split('\\').pop() || activeMesh3dStudio?.name || 'Model Loaded'
                        : activeMesh3dStudio?.name || 'No Model Loaded'}
                    </span>
                    <span className="model-meta">
                      {activeMesh3dLoaded ? 'Ready for 3D mesh generation' : activeMesh3dStudio ? 'Not currently loaded — load it to use this studio' : 'Use the sidebar to configure and load a model'}
                    </span>
                  </div>
                </div>
              </div>

              <h2 style={{ fontSize: '1.4rem', marginBottom: '0.3rem' }}>3D Model Studio</h2>
              <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '1.5rem' }}>
                Local text / image / multi-image → 3D mesh generation
              </p>

              <div className="grid-2">
                <div className="card">
                  {mesh3dAvailableKinds.length > 1 && (
                    <div className="segmented-control">
                      {mesh3dAvailableKinds.map(kind => (
                        <button
                          key={kind}
                          type="button"
                          className={`segmented-btn ${mesh3dInputKind === kind ? 'active' : ''}`}
                          onClick={() => setMesh3dInputKind(kind)}
                        >
                          {kind === 'text' ? 'Text' : kind === 'image' ? 'Image' : 'Multi-Image'}
                        </button>
                      ))}
                    </div>
                  )}

                  {mesh3dInputKind === 'text' && (
                    <div className="form-group">
                      <label>Prompt <span className="field-required-badge">Required</span></label>
                      <textarea
                        rows={4}
                        style={{ height: 'auto' }}
                        value={mesh3dPrompt}
                        onChange={e => setMesh3dPrompt(e.target.value)}
                      />
                    </div>
                  )}

                  {mesh3dInputKind === 'image' && (
                    <div className="form-group">
                      <label>Source Image <span className="field-required-badge">Required</span></label>
                      <input
                        ref={mesh3dImageInputRef}
                        type="file"
                        accept="image/*"
                        style={{ display: 'none' }}
                        onChange={e => { handleMesh3dSingleImage(e.target.files); e.target.value = ''; }}
                      />
                      {mesh3dImages[0] ? (
                        <div className="mesh3d-dropzone-thumb">
                          <img src={mesh3dImages[0].dataUrl} alt={mesh3dImages[0].name} />
                          <button type="button" className="mesh3d-thumb-remove" onClick={() => setMesh3dImages([])} title="Remove"><X size={12} /></button>
                          <button type="button" className="mesh3d-dropzone-replace" onClick={() => mesh3dImageInputRef.current?.click()}>Replace</button>
                        </div>
                      ) : (
                        <div className="mesh3d-dropzone" onClick={() => mesh3dImageInputRef.current?.click()}>
                          <Paperclip size={20} />
                          <span>Click to upload an image</span>
                        </div>
                      )}
                    </div>
                  )}

                  {mesh3dInputKind === 'multi_image' && (
                    <div className="form-group">
                      <label>Source Images <span className="field-required-badge">Required</span></label>
                      <input
                        ref={mesh3dMultiImageInputRef}
                        type="file"
                        accept="image/*"
                        multiple
                        style={{ display: 'none' }}
                        onChange={e => { handleMesh3dMultiImages(e.target.files); e.target.value = ''; }}
                      />
                      <div className="mesh3d-thumb-grid">
                        {mesh3dImages.map((img, idx) => (
                          <div key={idx} className="mesh3d-thumb-grid-item">
                            <img src={img.dataUrl} alt={img.name} />
                            <button type="button" className="mesh3d-thumb-remove" onClick={() => setMesh3dImages(current => current.filter((_, i) => i !== idx))} title="Remove"><X size={12} /></button>
                          </div>
                        ))}
                        <div className="mesh3d-dropzone mesh3d-dropzone-add" onClick={() => mesh3dMultiImageInputRef.current?.click()}>
                          <Paperclip size={18} />
                          <span>Add images</span>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Per-adapter tunable params, rendered generically from
                      GET /v1/models3d/schema — see mesh3dAdapterParams. */}
                  {mesh3dAdapterParams.map(p => (
                    <SchemaParamField
                      key={p.name}
                      param={p}
                      value={mesh3dParamValues[p.name]}
                      onChange={val => setMesh3dParam(p.name, val)}
                    />
                  ))}
                  {!mesh3dSchema && (
                    <p className="slider-hint" style={{ marginBottom: '1rem' }}>
                      Model-specific parameters are unavailable (could not reach the daemon's schema endpoint). Generation will use each backend's own defaults.
                    </p>
                  )}

                  <div className="form-group">
                    <label>Output Format</label>
                    <select value={mesh3dFormat} onChange={e => setMesh3dFormat(e.target.value)}>
                      <option value="glb">GLB</option>
                      <option value="obj">OBJ</option>
                      <option value="ply">PLY</option>
                      <option value="stl">STL</option>
                      <option value="fbx">FBX</option>
                    </select>
                  </div>

                  <div className="slider-header" style={{ marginBottom: '1rem' }}>
                    <div><strong>Texture</strong><div className="slider-hint">Generate a texture/material for the mesh.</div></div>
                    <input type="checkbox" checked={mesh3dTexture} onChange={e => setMesh3dTexture(e.target.checked)} />
                  </div>

                  {mesh3dMissingField && (
                    <p className="field-validation-warning">
                      {mesh3dMissingField === 'prompt' ? 'Enter a prompt before generating.' : 'Attach a source image before generating.'}
                    </p>
                  )}
                  <div className="gen-btn-row">
                    <button className="btn-primary" onClick={handleGenerateMesh3d} disabled={isGeneratingMesh || Boolean(mesh3dMissingField)}>
                      <Boxes size={16} /> {isGeneratingMesh ? 'Generating...' : 'Generate 3D Model'}
                    </button>
                    {isGeneratingMesh && (
                      <button className="btn-cancel-gen" onClick={() => cancelGenerationJob(activeMeshJobId)}>
                        <X size={16} /> Cancel
                      </button>
                    )}
                  </div>
                  {isGeneratingMesh && <GenProgressBar progress={meshProgress} />}
                  {installProgress && <GenProgressBar progress={installProgress} />}
                </div>

                <div className="card">
                  <label>Generated Mesh Output</label>
                  {mesh3dResult ? (
                    <Mesh3DViewer base64={mesh3dResult.base64} url={mesh3dResult.url} format={mesh3dResult.format} />
                  ) : (
                    <div className="preview-box mesh3d-empty-preview">
                      <Boxes size={40} />
                      <span>Generated 3D mesh will display here</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

        </main>
      </div>

      {/* Configure Model Sidebar (left overlay) */}
      {configTarget && (
        <>
          <div className="config-sidebar-backdrop" onClick={closeConfigPanel} />
          <aside className="config-sidebar">
            <div className="config-sidebar-header">
              <button className="config-sidebar-close" onClick={closeConfigPanel} title="Close">
                <X size={16} />
              </button>
              <div className="config-sidebar-title">
                <Settings size={16} /> {configTarget.name || 'Model'} settings
              </div>
            </div>
            <div className="config-sidebar-body">
              <div className="sidebar-section">
                <label className="section-label">Model path</label>
                <input className="control-input" value={configTarget?.model_path || ''} onChange={e => setConfigTarget(current => ({ ...current, model_path: e.target.value }))} />
              </div>
              {cfgCategory === 'chat' && (
                <div className="sidebar-section">
                  <label className="section-label">Vision projector (mmproj)</label>
                  <div className="mmproj-row">
                    <input className="control-input" value={configTarget?.mmproj_path || ''} onChange={e => setConfigTarget(current => ({ ...current, mmproj_path: e.target.value }))} />
                    <button className="btn-browse" onClick={async () => {
                      const defaultPath = configTarget?.model_path ? window.require('path').dirname(configTarget.model_path) : undefined;
                      const filePath = await browseForFile(defaultPath, [{ name: 'GGUF', extensions: ['gguf'] }]);
                      if (filePath) setConfigTarget(current => ({ ...current, mmproj_path: filePath }));
                    }}>Browse</button>
                  </div>
                  <div className="slider-hint">Auto-detected from the model directory. Leave empty if this is not a vision model.</div>
                </div>
              )}
              {cfgCategory === 'chat' && (
                <div className="sidebar-section">
                  <label className="section-label">MTP drafter (multi-token prediction)</label>
                  <div className="mmproj-row">
                    <input className="control-input" value={configTarget?.mtp_path || ''} onChange={e => setConfigTarget(current => ({ ...current, mtp_path: e.target.value }))} />
                    <button className="btn-browse" onClick={async () => {
                      const defaultPath = configTarget?.model_path ? window.require('path').dirname(configTarget.model_path) : undefined;
                      const filePath = await browseForFile(defaultPath, [{ name: 'GGUF', extensions: ['gguf'] }]);
                      if (filePath) setConfigTarget(current => ({ ...current, mtp_path: filePath }));
                    }}>Browse</button>
                  </div>
                  <div className="slider-hint">Auto-detected from the model directory. Leave empty if this model has no drafter.</div>
                </div>
              )}
              {cfgCategory === 'chat' && (
                <div className="sidebar-section">
                  <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                    <input type="checkbox" checked={configTarget?.mtp_enabled ?? true} onChange={e => setConfigTarget(current => ({ ...current, mtp_enabled: e.target.checked }))} />
                    <span className="section-label" style={{ margin: 0 }}>Enable MTP speculative decoding</span>
                  </label>
                </div>
              )}
              {cfgCategory === 'chat' && (
                <div className="sidebar-section">
                  <label className="section-label">Draft tokens per step</label>
                  <div className="mmproj-row">
                    <input
                      className="control-input"
                      type="number"
                      min="1"
                      max="16"
                      placeholder="4"
                      value={configTarget?.spec_draft_n_max ?? ''}
                      onChange={e => {
                        const raw = e.target.value;
                        const parsed = raw === '' ? undefined : parseInt(raw, 10);
                        setConfigTarget(current => ({ ...current, spec_draft_n_max: Number.isNaN(parsed) ? undefined : parsed }));
                      }}
                    />
                  </div>
                  <div className="slider-hint">Max draft tokens per MTP step. Defaults to 4 when empty.</div>
                </div>
              )}
              {cfgCategory === 'chat' && (
                <div className="sidebar-section">
                  <label className="section-label">Min acceptance probability</label>
                  <div className="mmproj-row">
                    <input
                      className="control-input"
                      type="number"
                      min="0"
                      max="1"
                      step="0.05"
                      placeholder="0.00"
                      value={configTarget?.spec_draft_p_min ?? ''}
                      onChange={e => {
                        const raw = e.target.value;
                        const parsed = raw === '' ? undefined : parseFloat(raw);
                        setConfigTarget(current => ({ ...current, spec_draft_p_min: Number.isNaN(parsed) ? undefined : parsed }));
                      }}
                    />
                  </div>
                  <div className="slider-hint">Minimum acceptance probability for draft tokens. Defaults to 0.75 when empty.</div>
                </div>
              )}
              {cfgCategory === 'chat' && (
                <>
                  <div className="sidebar-section">
                    <div className="slider-header"><label className="section-label">GPU layers</label><span className="badge-value">{configTarget?.gpu_layers ?? 99}</span></div>
                    <input className="control-slider" type="range" min="0" max="99" value={configTarget?.gpu_layers ?? 99} onChange={e => setConfigTarget(current => ({ ...current, gpu_layers: Number(e.target.value) }))} />
                  </div>
                  <div className="sidebar-section">
                    <div className="slider-header"><label className="section-label">Context window</label><span className="badge-value">{(configTarget?.context_size ?? 4096).toLocaleString()} / {configTarget?.max_context_size ? configTarget.max_context_size.toLocaleString() : '?'}</span></div>
                    {configTarget?.max_context_size ? <>
                      <input className="control-slider" type="range" min="512" max={configTarget.max_context_size} step="512" value={Math.min(configTarget.context_size ?? 4096, configTarget.max_context_size)} onChange={e => setConfigTarget(current => ({ ...current, context_size: Number(e.target.value) }))} />
                      <div className="slider-hint">Maximum read from this GGUF model’s metadata.</div>
                    </> : <div className="slider-hint">Maximum context is not present in this model’s metadata.</div>}
                  </div>
                  <div className="sidebar-section">
                    <div className="slider-header"><label className="section-label">Temperature</label><span className="badge-value">{configTarget?.temperature ?? 0.7}</span></div>
                    <input className="control-slider" type="range" min="0" max="1.5" step="0.05" value={configTarget?.temperature ?? 0.7} onChange={e => setConfigTarget(current => ({ ...current, temperature: Number(e.target.value) }))} />
                  </div>
                  <div className="sidebar-section">
                    <div className="slider-header"><label className="section-label">Top-P</label><span className="badge-value">{configTarget?.top_p ?? 0.9}</span></div>
                    <input className="control-slider" type="range" min="0.1" max="1" step="0.05" value={configTarget?.top_p ?? 0.9} onChange={e => setConfigTarget(current => ({ ...current, top_p: Number(e.target.value) }))} />
                  </div>
                  <div className="sidebar-section">
                    <div className="slider-header">
                      <label className="section-label">System Prompt</label>
                      <div className="mmproj-row">
                        <button className="btn-browse" onClick={() => setConfigTarget(current => ({ ...current, system_prompt: '' }))}>Reset</button>
                      </div>
                    </div>
                    <textarea
                      className="control-input"
                      rows={4}
                      value={configTarget?.system_prompt || ''}
                      onChange={e => setConfigTarget(current => ({ ...current, system_prompt: e.target.value }))}
                      placeholder="Leave empty to use the app's default system prompt."
                    />
                    <div className="slider-hint">Sent with every chat request to this model. Reset clears it back to the app default.</div>
                  </div>
                </>
              )}
              {cfgCategory === 'image' && (
                <>
                  <div className="sidebar-section">
                    <div className="slider-header"><label className="section-label">Sampling steps</label><span className="badge-value">{configTarget?.steps ?? 25}</span></div>
                    <input className="control-slider" type="range" min="1" max="50" step="1" value={configTarget?.steps ?? 25} onChange={e => setConfigTarget(current => ({ ...current, steps: Number(e.target.value) }))} />
                    <div className="slider-hint">Default denoising steps for generation.</div>
                  </div>
                  <div className="sidebar-section">
                    <div className="slider-header"><label className="section-label">CFG scale</label><span className="badge-value">{configTarget?.cfg_scale ?? 7}</span></div>
                    <input className="control-slider" type="range" min="1" max="15" step="0.5" value={configTarget?.cfg_scale ?? 7} onChange={e => setConfigTarget(current => ({ ...current, cfg_scale: Number(e.target.value) }))} />
                    <div className="slider-hint">How strongly the image follows the prompt.</div>
                  </div>
                  <div className="sidebar-section">
                    <label className="section-label">Default size</label>
                    <div className="mmproj-row">
                      <input className="control-input" type="number" min="64" step="64" value={configTarget?.width ?? 512} onChange={e => setConfigTarget(current => ({ ...current, width: Number(e.target.value) }))} />
                      <span style={{ color: 'var(--text-secondary)' }}>×</span>
                      <input className="control-input" type="number" min="64" step="64" value={configTarget?.height ?? 512} onChange={e => setConfigTarget(current => ({ ...current, height: Number(e.target.value) }))} />
                    </div>
                  </div>
                  <div className="sidebar-section">
                    <label className="section-label">Seed</label>
                    <input className="control-input" type="number" value={configTarget?.seed ?? -1} onChange={e => setConfigTarget(current => ({ ...current, seed: Number(e.target.value) }))} />
                    <div className="slider-hint">-1 draws a random seed each generation.</div>
                  </div>
                  {configTarget?.image_components?.length ? (
                    <div className="sidebar-section">
                      <label className="section-label">Sibling components</label>
                      {configTarget.image_components.map((comp, idx) => {
                        const download = comp.source ? hfDownloads[`${comp.source.repo}::${comp.source.filename}`] : null;
                        const progressPct = download && download.total_bytes > 0
                          ? Math.min(100, (download.downloaded_bytes / download.total_bytes) * 100)
                          : 0;
                        const assignStatus = componentAssignStatus[comp.target_path];
                        return (
                          <div key={idx} className="hf-file-row" style={{ paddingLeft: 0 }}>
                            <div className="hf-file-info">
                              <div className="hf-file-name-row">
                                <span className="hf-file-name">{comp.kind_name}</span>
                              </div>
                              {comp.resolved_path ? (
                                <span className="modal-list-item-meta">Found: {comp.resolved_path.split('\\').pop()}</span>
                              ) : comp.source ? (
                                <>
                                  {(!download || download.status === 'error') && (
                                    <button
                                      className="hf-download-btn"
                                      onClick={() => startHfDownload(comp.source.repo, comp.source.filename, comp.target_path, comp.source.target_filename)}
                                    >
                                      Download {comp.source.filename} ({comp.source.repo})
                                    </button>
                                  )}
                                  {download && download.status === 'downloading' && (
                                    <>
                                      <div className="hf-progress-bar">
                                        <div className="hf-progress-fill" style={{ width: `${progressPct}%` }} />
                                      </div>
                                      <div className="hf-progress-row">
                                        <span className="hf-progress-text">
                                          {formatFileSize(download.downloaded_bytes)} / {formatFileSize(download.total_bytes)}
                                        </span>
                                      </div>
                                    </>
                                  )}
                                  {download && download.status === 'complete' && (
                                    <span className="modal-list-item-meta">Downloaded</span>
                                  )}
                                  {download && download.status === 'error' && (
                                    <span className="modal-list-item-meta" style={{ color: '#ef4444' }}>
                                      Download failed{download.error ? `: ${download.error}` : ''}
                                    </span>
                                  )}
                                </>
                              ) : (
                                <span className="modal-list-item-meta">
                                  Missing — place a file manually at: {comp.target_path}
                                </span>
                              )}
                              <div className="mmproj-row" style={{ marginTop: '0.25rem' }}>
                                <button
                                  className="btn-browse"
                                  disabled={assignStatus?.status === 'assigning'}
                                  onClick={() => assignImageComponentOverride(comp)}
                                >
                                  {comp.resolved_path ? 'Change file...' : 'Assign file...'}
                                </button>
                                {assignStatus?.status === 'assigning' && (
                                  <span className="modal-list-item-meta">Assigning...</span>
                                )}
                                {assignStatus?.status === 'error' && (
                                  <span className="modal-list-item-meta" style={{ color: '#ef4444' }}>{assignStatus.error}</span>
                                )}
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  ) : null}
                </>
              )}
              {cfgCategory === 'tts' && (
                <div className="sidebar-section">
                  <div className="slider-header"><label className="section-label">Speech speed</label><span className="badge-value">{(configTarget?.speed ?? 1).toFixed(2)}×</span></div>
                  <input className="control-slider" type="range" min="0.5" max="2" step="0.05" value={configTarget?.speed ?? 1} onChange={e => setConfigTarget(current => ({ ...current, speed: Number(e.target.value) }))} />
                  <div className="slider-hint">Playback rate for synthesized speech. Voice is chosen per request in the studio.</div>
                </div>
              )}
              {cfgCategory === 'mesh3d' && (
                <>
                  {configTarget?.mesh_components?.length ? configTarget.mesh_components.map((component, index) => (
                    <div className="sidebar-section" key={`${component.label}-${index}`}>
                      <label className="section-label">{component.label}</label>
                      <div className="mmproj-row">
                        <input
                          className="control-input"
                          value={component.path || ''}
                          onChange={e => setConfigTarget(current => {
                            const next = [...current.mesh_components];
                            next[index] = { ...next[index], path: e.target.value };
                            return { ...current, mesh_components: next };
                          })}
                        />
                        <button className="btn-browse" onClick={async () => {
                          const defaultPath = configTarget?.model_path ? window.require('path').dirname(configTarget.model_path) : undefined;
                          const filePath = await browseForFile(defaultPath, undefined, ['openDirectory']);
                          if (!filePath) return;
                          setConfigTarget(current => {
                            const next = [...current.mesh_components];
                            next[index] = { ...next[index], path: filePath };
                            return { ...current, mesh_components: next };
                          });
                        }}>Browse</button>
                      </div>
                      <div className="slider-hint">Auto-detected from model directory. Override if needed.</div>
                    </div>
                  )) : (
                    <div className="sidebar-section">
                      <div className="slider-hint">No sibling components detected for this model.</div>
                    </div>
                  )}
                  {configTarget?.mesh_input_kinds?.length ? (
                    <div className="sidebar-section">
                      <label className="section-label">Accepted inputs</label>
                      <div className="slider-hint">{configTarget.mesh_input_kinds.map(k => k.replace('_', ' ')).join(', ')}</div>
                    </div>
                  ) : null}
                  <div className="sidebar-section">
                    <div className="slider-header"><label className="section-label">Sampling steps</label><span className="badge-value">{configTarget?.steps ?? 64}</span></div>
                    <input className="control-slider" type="range" min="1" max="64" step="1" value={configTarget?.steps ?? 64} onChange={e => setConfigTarget(current => ({ ...current, steps: Number(e.target.value) }))} />
                  </div>
                  <div className="sidebar-section">
                    <div className="slider-header"><label className="section-label">Guidance scale</label><span className="badge-value">{configTarget?.guidance_scale ?? 15}</span></div>
                    <input className="control-slider" type="range" min="0" max="30" step="0.5" value={configTarget?.guidance_scale ?? 15} onChange={e => setConfigTarget(current => ({ ...current, guidance_scale: Number(e.target.value) }))} />
                  </div>
                  <div className="sidebar-section">
                    <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                      <input type="checkbox" checked={configTarget?.texture ?? true} onChange={e => setConfigTarget(current => ({ ...current, texture: e.target.checked }))} />
                      <span className="section-label" style={{ margin: 0 }}>Generate texture</span>
                    </label>
                  </div>
                  <div className="sidebar-section">
                    <label className="section-label">Seed</label>
                    <input className="control-input" type="number" value={configTarget?.seed ?? -1} onChange={e => setConfigTarget(current => ({ ...current, seed: Number(e.target.value) }))} />
                    <div className="slider-hint">-1 draws a random seed each generation.</div>
                  </div>
                </>
              )}
              {cfgCategory === 'audio' && (
                <div className="sidebar-section">
                  <div className="slider-hint">Speech-to-text model. Transcription options are chosen per request in the studio.</div>
                </div>
              )}
              {cfgCategory === 'video' && (
                <div className="sidebar-section">
                  <label className="section-label">Text Encoder</label>
                  <div className="mmproj-row">
                    <input className="control-input" value={configTarget?.text_encoder_override_path || ''} onChange={e => setConfigTarget(current => ({ ...current, text_encoder_override_path: e.target.value }))} />
                    <button className="btn-browse" onClick={async () => {
                      const defaultPath = configTarget?.model_path ? window.require('path').dirname(configTarget.model_path) : undefined;
                      const filePath = await browseForFile(defaultPath, [{ name: 'Text Encoder', extensions: ['gguf', 'safetensors', 'ckpt', 'pt', 'bin'] }]);
                      if (filePath) setConfigTarget(current => ({ ...current, text_encoder_override_path: filePath }));
                    }}>Browse</button>
                  </div>
                  <div className="slider-hint">Overrides auto-detection. Leave empty to use the model's own text encoder / sibling files.</div>
                </div>
              )}
              {cfgCategory === 'video' && (
                <div className="sidebar-section">
                  <label className="section-label">VAE</label>
                  <div className="mmproj-row">
                    <input className="control-input" value={configTarget?.vae_override_path || ''} onChange={e => setConfigTarget(current => ({ ...current, vae_override_path: e.target.value }))} />
                    <button className="btn-browse" onClick={async () => {
                      const defaultPath = configTarget?.model_path ? window.require('path').dirname(configTarget.model_path) : undefined;
                      const filePath = await browseForFile(defaultPath, [{ name: 'VAE', extensions: ['gguf', 'safetensors', 'ckpt', 'pt', 'bin'] }]);
                      if (filePath) setConfigTarget(current => ({ ...current, vae_override_path: filePath }));
                    }}>Browse</button>
                  </div>
                  <div className="slider-hint">Overrides auto-detection. Leave empty to use the model's own VAE / sibling files.</div>
                </div>
              )}
              {cfgCategory === 'video' && (
                <div className="sidebar-section">
                  <div className="slider-hint">Video model. Generation parameters are chosen per request in the studio.</div>
                  {configTarget?.vae_path ? (
                    <div className="modal-list-item-meta" style={{ marginTop: '0.5rem' }}>
                      Sibling VAE found: {configTarget.vae_path.split('\\').pop()}
                    </div>
                  ) : null}
                </div>
              )}
              {cfgCategory === 'video' && configTarget?.video_components?.length ? (
                <div className="sidebar-section">
                  <label className="section-label">Pipeline components</label>
                  <div className="slider-hint">
                    Needed to run this model. If missing, they would otherwise be downloaded silently the first time you generate.
                  </div>
                  {Object.entries(
                    configTarget.video_components.reduce((groups, comp) => {
                      (groups[comp.group] ||= []).push(comp);
                      return groups;
                    }, {})
                  ).map(([group, comps]) => {
                    const allResolved = comps.every(c => c.resolved_path);
                    return (
                      <div key={group} style={{ marginTop: '0.5rem' }}>
                        <div className="hf-file-name-row">
                          <span className="hf-file-name">{group}</span>
                          <span className="modal-list-item-meta">{allResolved ? 'Ready' : `${comps.filter(c => c.resolved_path).length}/${comps.length} downloaded`}</span>
                        </div>
                        {!allResolved && (
                          <button
                            className="hf-download-btn"
                            onClick={() => comps.forEach(c => {
                              if (!c.resolved_path && c.source) startHfDownload(c.source.repo, c.source.filename, c.target_path, c.source.target_filename);
                            })}
                          >
                            Download {group}
                          </button>
                        )}
                        {comps.filter(c => !c.resolved_path && c.source).map((c, idx) => {
                          const download = hfDownloads[`${c.source.repo}::${c.source.filename}`];
                          const progressPct = download && download.total_bytes > 0
                            ? Math.min(100, (download.downloaded_bytes / download.total_bytes) * 100)
                            : 0;
                          return download && download.status !== 'complete' ? (
                            <div key={idx} className="modal-list-item-meta">
                              {c.kind_name}: {download.status === 'error' ? 'Error' : `${progressPct.toFixed(0)}%`}
                            </div>
                          ) : null;
                        })}
                        {comps.filter(c => !c.resolved_path).map((c, idx) => {
                          const assignStatus = componentAssignStatus[c.target_path];
                          return (
                            <div key={`assign-${idx}`} className="mmproj-row" style={{ marginTop: '0.25rem' }}>
                              <span className="modal-list-item-meta">{c.kind_name}:</span>
                              <button
                                className="btn-browse"
                                disabled={assignStatus?.status === 'assigning'}
                                onClick={() => assignComponentFile(c)}
                              >
                                Assign file...
                              </button>
                              {assignStatus?.status === 'assigning' && (
                                <span className="modal-list-item-meta">Assigning...</span>
                              )}
                              {assignStatus?.status === 'error' && (
                                <span className="modal-list-item-meta" style={{ color: '#ef4444' }}>{assignStatus.error}</span>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    );
                  })}
                </div>
              ) : null}
              <div className="vram-preview-card">
                <div className="preview-title"><Cpu size={13} /> Estimated VRAM</div>
                <strong>{((modelFitPreview?.total_required_vram_bytes || 0) / 1e9).toFixed(2)} GB</strong>
              </div>
              <button className="btn-load-model config-sidebar-load" onClick={async () => {
                const cfg = configTarget;
                const settings = buildSettingsFromConfigTarget(cfg);
                const card = modelCards.find(c => c.id === cfg?.cardId);
                if (card) {
                  await loadCardModel(card, settings);
                } else {
                  const modelId = await handleLoadModel({ model_path: cfg?.model_path, gpu_layers: cfg?.gpu_layers ?? 99, context_size: cfg?.context_size ?? 4096, mmproj_path: cfg?.mmproj_path || undefined, mtp_path: cfg?.mtp_path || undefined, mtp_enabled: cfg?.mtp_enabled, spec_draft_n_max: cfg?.spec_draft_n_max, spec_draft_p_min: cfg?.spec_draft_p_min, text_encoder_override_path: cfg?.text_encoder_override_path || undefined, vae_override_path: cfg?.vae_override_path || undefined });
                  if (modelId) applyModelDefaults(cfg);
                }
              }} disabled={isLoadingModel}>
                <Zap size={15} /> {isLoadingModel ? 'Loading...' : 'Load configured model'}
              </button>
              <div className="config-sidebar-section-title">Loaded models ({loadedModels.length})</div>
              {loadedModels.length === 0 ? <p style={{ color: 'var(--text-secondary)', fontSize: '0.8rem' }}>No models are loaded.</p> : loadedModels.map(model => (
                <div key={model.model_id} className="config-sidebar-loaded-row">
                  <strong>{model.model_path?.split('\\').pop() || model.model_id}</strong>
                  <div className="config-sidebar-loaded-meta">
                    {model.gpu_layers ?? 0} GPU layers · {model.context_size ?? 0} context
                  </div>
                  <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
                    <button className="btn-load-model" onClick={() => setSelectedModelId(model.model_id)} disabled={selectedModelId === model.model_id}>
                      {selectedModelId === model.model_id ? 'Selected for chat' : 'Use for chat'}
                    </button>
                    <button className="btn-unload-model" onClick={() => handleUnloadModel(model.model_id)} disabled={unloadingModelId === model.model_id}>
                      <Power size={14} /> Eject
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </aside>
        </>
      )}
      <ErrorToasts />
    </div>
  );
}

export default function App() {
  return (
    <ErrorLogProvider>
      <AppInner />
    </ErrorLogProvider>
  );
}
