import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  CheckCircle2, XCircle, Loader, RefreshCw, MessageSquare,
  Image, Video, Box, Volume2, Sparkles,
} from 'lucide-react';

// Modality checkboxes shown in Step 1. `id` doubles as the key sent to
// startSetup() and as a keyword used to match progress phases to steps below.
const MODALITY_OPTIONS = [
  { id: 'image', label: 'Image generation', icon: Image },
  { id: 'video', label: 'Video generation', icon: Video },
  { id: 'threed', label: '3D generation', icon: Box },
  { id: 'voice', label: 'Voice / TTS', icon: Volume2 },
];

const BYTES_GB = 1024 * 1024 * 1024;

function formatBytes(bytes) {
  if (!bytes && bytes !== 0) return '';
  if (bytes >= BYTES_GB) return `${(bytes / BYTES_GB).toFixed(1)} GB`;
  return `${(bytes / (1024 * 1024)).toFixed(0)} MB`;
}

// Keyword sets used to route an incoming `{ phase }` progress event to the
// right step row. The exact phase strings emitted by download-binaries.mjs /
// the Python setup scripts aren't finalized yet, so this matches loosely
// (substring, case-insensitive) rather than requiring an exact phase name.
const STEP_PHASE_KEYWORDS = {
  binaries: ['llama-server', 'llama_server', 'sd', 'binaries', 'binary'],
  uv: ['uv'],
  image: ['image', 'diffusers', 'sd-env', 'sd_env'],
  video: ['video'],
  threed: ['3d', 'threed', 'mesh', 'shap'],
  voice: ['voice', 'tts', 'kokoro'],
};

function phaseMatchesStep(phase, stepId) {
  if (!phase) return false;
  const p = String(phase).toLowerCase();
  const keywords = STEP_PHASE_KEYWORDS[stepId] || [];
  return keywords.some((kw) => p.includes(kw));
}

function buildSteps(selected) {
  const steps = [
    { id: 'binaries', label: 'Native binaries' },
    { id: 'uv', label: 'uv (Python installer)' },
  ];
  for (const opt of MODALITY_OPTIONS) {
    if (selected[opt.id]) {
      steps.push({ id: opt.id, label: `${opt.label} environment` });
    }
  }
  return steps.map((s) => ({
    ...s,
    status: 'pending', // pending | running | done | error
    message: null,
    bytesDownloaded: null,
    totalBytes: null,
  }));
}

function StatusIcon({ status }) {
  if (status === 'done') return <CheckCircle2 size={18} className="wiz-status-icon done" />;
  if (status === 'error') return <XCircle size={18} className="wiz-status-icon error" />;
  if (status === 'running') return <Loader size={18} className="wiz-status-icon running spin" />;
  return <div className="wiz-status-dot pending" />;
}

function formatStepStatusText(step) {
  const parts = [];
  if (step.message) parts.push(step.message);
  if (step.status === 'running' && step.totalBytes && step.bytesDownloaded != null) {
    parts.push(`${formatBytes(step.bytesDownloaded)} / ${formatBytes(step.totalBytes)}`);
  }
  return parts.join(' — ');
}

function StepProgressBar({ step }) {
  if (step.status === 'pending') return <div className="wiz-progress-track wiz-progress-empty" />;
  if (step.status === 'done') return <div className="wiz-progress-track"><div className="wiz-progress-fill done" style={{ width: '100%' }} /></div>;
  if (step.status === 'error') return <div className="wiz-progress-track"><div className="wiz-progress-fill error" style={{ width: '100%' }} /></div>;
  // running
  if (step.totalBytes && step.bytesDownloaded != null) {
    const pct = Math.min(100, Math.round((step.bytesDownloaded / step.totalBytes) * 100));
    return (
      <div className="wiz-progress-track">
        <div className="wiz-progress-fill running" style={{ width: `${pct}%` }} />
      </div>
    );
  }
  return <div className="wiz-progress-track wiz-progress-indeterminate"><div className="wiz-progress-fill running" /></div>;
}

export default function SetupWizard() {
  const [step, setStep] = useState(1);
  const [selected, setSelected] = useState({ image: false, video: false, threed: false, voice: false });
  const [installSteps, setInstallSteps] = useState([]);
  const sizes = useMemo(() => window.disposWizard?.getComponentSizes?.() || {}, []);

  const toggleModality = (id) => {
    setSelected((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const applyProgress = useCallback((data) => {
    if (!data) return;
    setInstallSteps((prev) => prev.map((s) => {
      if (!phaseMatchesStep(data.phase, s.id)) return s;
      return {
        ...s,
        status: data.status || s.status,
        message: data.message ?? s.message,
        bytesDownloaded: data.bytesDownloaded ?? s.bytesDownloaded,
        totalBytes: data.totalBytes ?? s.totalBytes,
      };
    }));
  }, []);

  useEffect(() => {
    if (!window.disposWizard) return;
    window.disposWizard.onProgress(applyProgress);
    window.disposWizard.onComplete(() => {
      setInstallSteps((prev) => {
        if (!prev.some((s) => s.status === 'error')) setStep(3);
        return prev;
      });
    });
  }, [applyProgress]);

  const handleContinue = () => {
    const steps = buildSteps(selected);
    setInstallSteps(steps);
    setStep(2);
    const selectedIds = Object.keys(selected).filter((id) => selected[id]);
    window.disposWizard?.startSetup?.(selectedIds);
  };

  const handleRetry = (stepId) => {
    setInstallSteps((prev) => prev.map((s) => (s.id === stepId ? { ...s, status: 'running', message: null } : s)));
    window.disposWizard?.retryStep?.(stepId);
  };

  const handleLaunch = () => {
    window.disposWizard?.launchMainApp?.();
  };

  return (
    <div className="wizard-shell">
      <div className="wizard-header">
        <Sparkles size={22} className="wizard-header-icon" />
        <div>
          <div className="wizard-title">Dispos Studio Setup</div>
          <div className="wizard-subtitle">
            {step === 1 && 'Choose the components you want to install'}
            {step === 2 && 'Installing selected components'}
            {step === 3 && 'Ready to go'}
          </div>
        </div>
      </div>

      {step === 1 && (
        <div className="wizard-body">
          <div className="wizard-component-list">
            <label className="wizard-component-row wizard-component-required">
              <input type="checkbox" checked disabled />
              <MessageSquare size={18} className="wizard-component-icon" />
              <div className="wizard-component-text">
                <div className="wizard-component-label">Chat</div>
                <div className="wizard-component-meta">{formatBytes(sizes.chat?.bytes || 200 * 1024 * 1024)}, required</div>
              </div>
            </label>
            {MODALITY_OPTIONS.map((opt) => {
              const Icon = opt.icon;
              const size = sizes[opt.id]?.bytes;
              return (
                <label key={opt.id} className="wizard-component-row">
                  <input
                    type="checkbox"
                    checked={!!selected[opt.id]}
                    onChange={() => toggleModality(opt.id)}
                  />
                  <Icon size={18} className="wizard-component-icon" />
                  <div className="wizard-component-text">
                    <div className="wizard-component-label">{opt.label}</div>
                    <div className="wizard-component-meta">{formatBytes(size)}</div>
                  </div>
                </label>
              );
            })}
          </div>
          <button className="wizard-btn-primary" onClick={handleContinue}>Continue</button>
        </div>
      )}

      {step === 2 && (
        <div className="wizard-body">
          <div className="wizard-step-list">
            {installSteps.map((s) => (
              <div key={s.id} className={`wizard-step-row wizard-step-${s.status}`}>
                <StatusIcon status={s.status} />
                <div className="wizard-step-main">
                  <div className="wizard-step-label">{s.label}</div>
                  <StepProgressBar step={s} />
                  {s.status === 'error' && s.message && (
                    <div className="wizard-step-error-msg">{s.message}</div>
                  )}
                  {s.status !== 'error' && s.status !== 'pending' && formatStepStatusText(s) && (
                    <div className="wizard-step-status-msg">{formatStepStatusText(s)}</div>
                  )}
                </div>
                {s.status === 'error' && (
                  <button className="wizard-btn-retry" onClick={() => handleRetry(s.id)}>
                    <RefreshCw size={14} /> Retry
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {step === 3 && (
        <div className="wizard-body wizard-body-center">
          <CheckCircle2 size={48} className="wizard-done-icon" />
          <div className="wizard-done-title">Setup complete</div>
          <button className="wizard-btn-primary" onClick={handleLaunch}>Launch DisposAI</button>
        </div>
      )}
    </div>
  );
}
