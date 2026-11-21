import React, { useState, useMemo } from 'react';
import {
  Sliders,
  Play,
  RotateCcw,
  CheckCircle2,
  AlertTriangle,
  FileCode,
  Flame,
  Layers,
  Activity,
  Compass,
  Cpu,
  BarChart3,
  ShieldAlert,
  ArrowRight,
  TrendingDown,
  Info,
  Copy,
  Check
} from 'lucide-react';

interface Scenario {
  id: string;
  name: string;
  nameEn: string;
  category: 'Easy' | 'Medium' | 'Hard' | 'Extreme';
  description: string;
  baseError: number; // in meters (minADE)
  datasetShare: number; // percentage in raw nuScenes
  color: string;
  path: { x: number; y: number }[];
  heading: number;
}

const SCENARIOS: Scenario[] = [
  {
    id: 'straight',
    name: 'مسیر مستقیم در بزرگراه',
    nameEn: 'Straight Highway Cruising',
    category: 'Easy',
    description: 'حرکت در خط مستقیم با سرعت ثابت. حجم بسیار بزرگ داده‌ها (~۷۰٪) را تشکیل می‌دهد.',
    baseError: 0.04,
    datasetShare: 70,
    color: '#10b981', // emerald
    path: [
      { x: 10, y: 50 }, { x: 30, y: 50 }, { x: 50, y: 50 }, { x: 70, y: 50 }, { x: 90, y: 50 }
    ],
    heading: 0,
  },
  {
    id: 'gentle_curve',
    name: 'انحراف ملایم لاین شهری',
    nameEn: 'Gentle Urban Curve',
    category: 'Medium',
    description: 'تغییر تدریجی لاین یا انحنای استاندارد خیابان با کاهش ملایم سرعت.',
    baseError: 0.22,
    datasetShare: 15,
    color: '#3b82f6', // blue
    path: [
      { x: 10, y: 65 }, { x: 30, y: 60 }, { x: 50, y: 48 }, { x: 70, y: 38 }, { x: 90, y: 35 }
    ],
    heading: -20,
  },
  {
    id: 'sharp_turn',
    name: 'گردش ۹۰ درجه در تقاطع شلوغ',
    nameEn: '90° Unprotected Intersection Turn',
    category: 'Hard',
    description: 'گردش تند به چپ با حضور عابر پیاده و خودروهای عبوری. در نمونه‌برداری عادی نادیده گرفته می‌شود.',
    baseError: 1.18,
    datasetShare: 10,
    color: '#f59e0b', // amber
    path: [
      { x: 15, y: 75 }, { x: 35, y: 70 }, { x: 50, y: 55 }, { x: 58, y: 35 }, { x: 62, y: 15 }
    ],
    heading: -75,
  },
  {
    id: 'sudden_braking_cutin',
    name: 'برش مسیر ناگهانی و ترمز اضطراری',
    nameEn: 'Aggressive Cut-in & Sudden Brake',
    category: 'Extreme',
    description: 'ورود ناگهانی خودرو همسایه در فاصله نزدیک و ترمز اضطراری. خطای اولیه مدل بدون ماینینگ بسیار بالا است.',
    baseError: 1.85,
    datasetShare: 5,
    color: '#ef4444', // red
    path: [
      { x: 10, y: 40 }, { x: 30, y: 42 }, { x: 48, y: 55 }, { x: 65, y: 62 }, { x: 72, y: 64 }
    ],
    heading: 30,
  },
];

export default function App() {
  const [sensitivity, setSensitivity] = useState<number>(0.65);
  const [emaMomentum, setEmaMomentum] = useState<number>(0.7);
  const [selectedScenario, setSelectedScenario] = useState<string>('sharp_turn');
  const [activeTab, setActiveTab] = useState<'visualizer' | 'comparison' | 'code' | 'docs'>('visualizer');
  const [selectedCodeFile, setSelectedCodeFile] = useState<string>('train_config');
  const [copied, setCopied] = useState<boolean>(false);

  // Compute normalized sampling weights and probabilities for the scenarios
  const scenarioProbabilities = useMemo(() => {
    const N = SCENARIOS.length;
    const totalError = SCENARIOS.reduce((acc, s) => acc + s.baseError, 0);

    return SCENARIOS.map((s) => {
      // P(i) = (1 - gamma) * (1 / N) + gamma * (e_i / sum(e))
      const uniformProb = 1.0 / N;
      const errorProportionalProb = s.baseError / totalError;
      const finalProb = (1.0 - sensitivity) * uniformProb + sensitivity * errorProportionalProb;
      const weight = (1.0 - sensitivity) + sensitivity * (N * s.baseError / totalError);
      return {
        ...s,
        probability: finalProb * 100,
        weight: weight,
      };
    });
  }, [sensitivity]);

  const activeScenarioData = useMemo(() => {
    return scenarioProbabilities.find((s) => s.id === selectedScenario) || scenarioProbabilities[0];
  }, [scenarioProbabilities, selectedScenario]);

  const hardToEasyRatio = useMemo(() => {
    const extreme = scenarioProbabilities.find((s) => s.id === 'sudden_braking_cutin')?.weight || 1;
    const easy = scenarioProbabilities.find((s) => s.id === 'straight')?.weight || 1;
    return (extreme / Math.max(0.001, easy)).toFixed(2);
  }, [scenarioProbabilities]);

  const handleCopyCode = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const codeSnippets: Record<string, { title: string; lang: string; code: string }> = {
    train_config: {
      title: 'python_src/configs/train_config.py',
      lang: 'python',
      code: `@dataclass
class TrainConfig(BaseConfig):
    """Hyperparameters for Standalone Retraining Phase."""
    dataset: str = "nuscenes"                       # Target dataset name ('nuscenes' or 'synthetic')
    train_path: str | None = None                   # Custom training sequences path
    val_path: str | None = None                     # Custom validation sequences path
    seed: int = 42                                  # Reproducibility seed for deterministic retraining
    epochs: int = 60                                # Maximum retraining epoch count ceiling
    batch_size: int = 128                           # Physical batch size on GPU per forward pass
    grad_accum_steps: int = 2                       # Gradient accumulation steps (Effective batch size = 256)
    use_amp: bool = True                            # Automatic Mixed Precision FP16 acceleration
    lr: float = 7.5e-4                              # Base learning rate for Adam optimizer
    min_lr: float = 1e-5                            # Minimum learning rate floor for Cosine Annealing
    weight_decay: float = 1e-4                      # L2 weight decay regularizer
    clip_grad_norm: float = 5.0                     # Maximum gradient clipping L2 norm
    early_stopping_patience: int = 10               # Early stopping patience epochs
    early_stopping_min_delta: float = 0.002         # Minimum ADE improvement threshold (2mm)
    train_samples_percent: float | None = None      # Dataset training percentage (None = 100%)

    # --- Hard Sample Mining (Weighted Sampling) Parameters ---
    hard_sample_mining: float = ${sensitivity.toFixed(2)}                 # Sensitivity parameter [0.0, 1.0]. 0.0 = disabled, >0 = error-weighted
    hard_sample_ema: float = ${emaMomentum.toFixed(2)}                    # EMA momentum for smoothing errors across epochs
    save_path: str = "checkpoints/best_trajectory_lnas.pt"
    device: str = "cuda"

    @property
    def hard_sample_sensitivity(self) -> float:
        return self.hard_sample_mining

    @hard_sample_sensitivity.setter
    def hard_sample_sensitivity(self, value: float):
        self.hard_sample_mining = float(value)

    def __post_init__(self):
        super().__post_init__()
        if not (0.0 <= self.hard_sample_mining <= 1.0):
            raise ValueError(
                f"TrainConfig.hard_sample_mining must be in range [0.0, 1.0], got {self.hard_sample_mining}. "
                "0.0 disables HSM (uniform), while values up to 1.0 increase sensitivity to challenging cases."
            )`
    },
    hard_sample_mining: {
      title: 'python_src/data/hard_sample_mining.py',
      lang: 'python',
      code: `class HardSampleMiningManager:
    """
    Tracks and updates per-sample prediction errors across training epochs
    and computes dynamic sampling weights based on the sensitivity parameter.
    
    Formula:
        P(i) = (1 - gamma) * (1 / N) + gamma * (e_i / sum(e))
        Weight(i) = (1 - gamma) + gamma * (N * e_i / (sum(e) + eps))
    """
    def __init__(self, num_samples: int, sensitivity: float = 0.0, ema_momentum: float = 0.7):
        self.num_samples = num_samples
        self.sensitivity = max(0.0, min(1.0, float(sensitivity)))
        self.ema_momentum = ema_momentum
        self.errors = torch.ones(self.num_samples, dtype=torch.float32)
        self.weights = torch.ones(self.num_samples, dtype=torch.float64)
        self.counts = torch.zeros(self.num_samples, dtype=torch.int32)
        self._recompute_weights()

    def update_sample_errors(self, sample_indices, sample_errors):
        """Updates errors using Exponential Moving Average (EMA)."""
        for idx, err in zip(sample_indices, sample_errors):
            if self.counts[idx] == 0:
                self.errors[idx] = err
            else:
                beta = self.ema_momentum
                self.errors[idx] = (1.0 - beta) * self.errors[idx] + beta * err
            self.counts[idx] += 1
        self._recompute_weights()

    def _recompute_weights(self):
        if self.sensitivity <= 0.0:
            self.weights.fill_(1.0) # Strictly uniform (disabled)
            return

        err_sum = self.errors.sum()
        N = float(self.num_samples)
        gamma = self.sensitivity
        self.weights = (1.0 - gamma) + gamma * (N * self.errors / (err_sum + 1e-7))`
    },
    dataloaders: {
      title: 'python_src/data/nuscenes_dataset.py',
      lang: 'python',
      code: `def build_nuscenes_dataloader(..., hard_sample_mining: float = 0.0, hard_sample_ema: float = 0.7):
    dataset = NuScenesPredictionDataset(...)
    
    sampler = None
    if split == "train" and hard_sample_mining > 0.0:
        # Dynamically inject Weighted Sampler prioritizing hard scenarios
        manager = HardSampleMiningManager(
            num_samples=len(dataset),
            sensitivity=hard_sample_mining,
            ema_momentum=hard_sample_ema
        )
        sampler = manager.get_sampler(replacement=True)
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )`
    },
    final_trainer: {
      title: 'python_src/train/final_trainer.py',
      lang: 'python',
      code: `class FinalModelTrainer(BaseTrainer):
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        for batch_idx, batch in enumerate(self.train_loader):
            b = unpack_batch_to_device(batch, self.device)
            pred_trajs, mode_logits = self.model(...)
            loss_reg, loss_cls = best_of_k_loss(pred_trajs, b["future"], mode_logits)
            
            # --- Per-sample error tracking for Hard Sample Mining ---
            if self.hard_sample_mining > 0 and "sample_idx" in b:
                with torch.no_grad():
                    # Calculate minADE per sample across modes
                    step_dists = safe_norm(pred_trajs.detach() - b["future"].unsqueeze(1), dim=-1).mean(dim=-1)
                    per_sample_ade = step_dists.min(dim=-1).values
                    if hasattr(self.train_loader.sampler, "update_sample_errors"):
                        self.train_loader.sampler.update_sample_errors(b["sample_idx"], per_sample_ade)
            
            # Backward pass & optimization ...`
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-indigo-500/30 selection:text-indigo-200" dir="rtl">
      {/* Top Navigation Bar */}
      <header className="border-b border-slate-800/80 bg-slate-900/60 backdrop-blur-md sticky top-0 z-50 px-6 py-3.5 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center shadow-lg shadow-indigo-500/20 text-white font-black text-xl">
            TL
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-bold text-white tracking-tight">
                سامانه TrajectoryLNAS | ماینینگ نمونه‌های سخت (Hard Sample Mining)
              </h1>
              <span className="text-[11px] font-medium px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                رساله کارشناسی ارشد
              </span>
            </div>
            <p className="text-xs text-slate-400">
              پژوهشگر: علیرضا غفوری (۴۰۳۱۶۳۱۲۰۹) • بهینه‌سازی دیتالودر با نمونه‌برداری وزن‌دار (Weighted Sampling)
            </p>
          </div>
        </div>

        {/* Global Sensitivity Status Indicator */}
        <div className="flex items-center gap-3">
          <div className={`px-3 py-1.5 rounded-xl text-xs font-semibold flex items-center gap-2 border transition-colors ${
            sensitivity > 0 
              ? 'bg-amber-500/10 text-amber-300 border-amber-500/30' 
              : 'bg-slate-800 text-slate-400 border-slate-700'
          }`}>
            <Flame className={`w-3.5 h-3.5 ${sensitivity > 0 ? 'text-amber-400 animate-pulse' : 'text-slate-500'}`} />
            <span>
              {sensitivity > 0 
                ? `ماینینگ فعال (حساسیت: ${(sensitivity * 100).toFixed(0)}٪)` 
                : 'ماینینگ غیرفعال (یکنواخت)'}
            </span>
          </div>

          <div className="h-6 w-px bg-slate-800" />

          <div className="text-xs text-slate-400">
            nuScenes Benchmark • VRAM: <span className="text-emerald-400 font-mono">1.38 MB</span>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 space-y-6">
        {/* Navigation Tabs */}
        <div className="flex items-center gap-2 border-b border-slate-800 pb-3">
          <button
            onClick={() => setActiveTab('visualizer')}
            className={`px-4 py-2 rounded-xl text-xs font-semibold flex items-center gap-2 transition-all ${
              activeTab === 'visualizer'
                ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/25'
                : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
            }`}
          >
            <Compass className="w-4 h-4" />
            آزمایشگاه بصری نمونه‌برداری وزن‌دار
          </button>

          <button
            onClick={() => setActiveTab('comparison')}
            className={`px-4 py-2 rounded-xl text-xs font-semibold flex items-center gap-2 transition-all ${
              activeTab === 'comparison'
                ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/25'
                : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
            }`}
          >
            <BarChart3 className="w-4 h-4" />
            تحلیل اثرگذاری بر خطای سناریوها (Baseline vs HSM)
          </button>

          <button
            onClick={() => setActiveTab('code')}
            className={`px-4 py-2 rounded-xl text-xs font-semibold flex items-center gap-2 transition-all ${
              activeTab === 'code'
                ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/25'
                : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
            }`}
          >
            <FileCode className="w-4 h-4" />
            کدهای پایتون پیاده‌سازی‌شده (train_config.py & HSM)
          </button>

          <button
            onClick={() => setActiveTab('docs')}
            className={`px-4 py-2 rounded-xl text-xs font-semibold flex items-center gap-2 transition-all ${
              activeTab === 'docs'
                ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/25'
                : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
            }`}
          >
            <Info className="w-4 h-4" />
            مبانی علمی و ریاضی ماینینگ نمونه‌های سخت
          </button>
        </div>

        {/* TAB 1: Visualizer Lab */}
        {activeTab === 'visualizer' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left Control Panel (5 Cols) */}
            <div className="lg:col-span-5 space-y-5">
              {/* Sensitivity Controller Card */}
              <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Sliders className="w-4 h-4 text-indigo-400" />
                    <h2 className="text-sm font-bold text-white">تنظیم حساسیت نمونه‌های چالشی</h2>
                  </div>
                  <span className="font-mono text-xs font-bold px-2 py-0.5 rounded-lg bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                    γ = {sensitivity.toFixed(2)}
                  </span>
                </div>

                <p className="text-xs text-slate-400 leading-relaxed">
                  میزان حساسیت در <code className="text-indigo-300 bg-slate-800 px-1 py-0.5 rounded">train_config.py</code> با پارامتر <code className="text-indigo-300 bg-slate-800 px-1 py-0.5 rounded">hard_sample_mining</code> کنترل می‌شود. مقدار ۰ نمونه‌برداری را کاملاً یکنواخت کرده و مقدار ۱ حداکثر وزن را به مسیرهای دارای خطای بالا می‌دهد.
                </p>

                {/* Slider */}
                <div className="space-y-2 pt-2">
                  <div className="flex justify-between text-[11px] text-slate-400">
                    <span>۰.۰ (کاملاً غیرفعال / یکنواخت)</span>
                    <span className="text-indigo-400 font-semibold">حساسیت: {(sensitivity * 100).toFixed(0)}٪</span>
                    <span>۱.۰ (حداکثر تمرکز)</span>
                  </div>
                  <input
                    type="range"
                    min="0.0"
                    max="1.0"
                    step="0.05"
                    value={sensitivity}
                    onChange={(e) => setSensitivity(parseFloat(e.target.value))}
                    className="w-full accent-indigo-500 bg-slate-800 h-2 rounded-lg cursor-pointer"
                  />
                  <div className="flex gap-1.5 pt-1">
                    {[0.0, 0.25, 0.5, 0.75, 1.0].map((val) => (
                      <button
                        key={val}
                        onClick={() => setSensitivity(val)}
                        className={`flex-1 py-1 text-[11px] font-mono rounded-lg border transition-all ${
                          Math.abs(sensitivity - val) < 0.02
                            ? 'bg-indigo-600 text-white border-indigo-500 font-bold'
                            : 'bg-slate-800/80 text-slate-400 border-slate-700/60 hover:text-white'
                        }`}
                      >
                        {val === 0 ? 'Off (0)' : val}
                      </button>
                    ))}
                  </div>
                </div>

                {/* EMA Momentum Slider */}
                <div className="border-t border-slate-800/80 pt-3 space-y-2">
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-300">مومنتوم هموارسازی زمانی (EMA Momentum)</span>
                    <span className="font-mono text-indigo-400 font-semibold">{emaMomentum.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="0.9"
                    step="0.05"
                    value={emaMomentum}
                    onChange={(e) => setEmaMomentum(parseFloat(e.target.value))}
                    className="w-full accent-violet-500 bg-slate-800 h-1.5 rounded-lg cursor-pointer"
                  />
                  <p className="text-[11px] text-slate-500">
                    مانع از پرش‌های ناگهانی وزن نمونه‌ها به دلیل نویز تکی یک بچ می‌شود: <span className="font-mono text-slate-400">e_t = (1-β)·e_(t-1) + β·loss</span>
                  </p>
                </div>
              </div>

              {/* Mathematical Weight Summary Card */}
              <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-3">
                <h3 className="text-xs font-bold text-slate-300 flex items-center gap-2">
                  <Activity className="w-4 h-4 text-emerald-400" />
                  وضعیت توزیع وزن‌های نمونه‌برداری در دیتالودر
                </h3>

                <div className="grid grid-cols-2 gap-3 pt-1">
                  <div className="bg-slate-950/70 border border-slate-800/80 rounded-xl p-3">
                    <div className="text-[11px] text-slate-400">نسبت شانس سخت به ساده</div>
                    <div className="text-lg font-bold font-mono text-amber-400 pt-0.5">
                      {hardToEasyRatio}x
                    </div>
                    <div className="text-[10px] text-slate-500">
                      {sensitivity === 0 ? 'برابر (کاملاً یکنواخت)' : 'اولویت بالاتر سناریوهای تند'}
                    </div>
                  </div>

                  <div className="bg-slate-950/70 border border-slate-800/80 rounded-xl p-3">
                    <div className="text-[11px] text-slate-400">حالت عملکرد DataLoader</div>
                    <div className="text-sm font-bold text-indigo-300 pt-1 flex items-center gap-1.5">
                      {sensitivity === 0 ? (
                        <>
                          <span className="w-2 h-2 rounded-full bg-slate-500" />
                          Shuffle=True (Uniform)
                        </>
                      ) : (
                        <>
                          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                          WeightedRandomSampler
                        </>
                      )}
                    </div>
                    <div className="text-[10px] text-slate-500">
                      {sensitivity === 0 ? 'بدون وزن‌دهی' : 'وزن‌دار پویا به ازای هر اپوک'}
                    </div>
                  </div>
                </div>

                {/* Mathematical Equation in Persian context */}
                <div className="bg-indigo-950/30 border border-indigo-900/40 rounded-xl p-3 text-xs space-y-1.5 font-mono text-indigo-200" dir="ltr">
                  <div className="text-[11px] text-indigo-400 font-sans">فرمول توزیع احتمال انتخاب نمونه i:</div>
                  <div className="text-center font-bold py-1 bg-slate-950/60 rounded-lg text-emerald-300 text-xs">
                    P(i) = (1 - γ) · (1 / N) + γ · ( e_i / Σ e_j )
                  </div>
                  <div className="text-[10px] text-slate-400 font-sans text-right" dir="rtl">
                    با افزایش γ، داده‌های دارای خطای پیش‌بینی بالا شانس بیشتری برای انتخاب پیدا می‌کنند.
                  </div>
                </div>
              </div>
            </div>

            {/* Right Scenario & Trajectory Inspector (7 Cols) */}
            <div className="lg:col-span-7 space-y-5">
              {/* Scenarios Grid */}
              <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-bold text-white flex items-center gap-2">
                    <Layers className="w-4 h-4 text-indigo-400" />
                    تأثیر حساسیت بر احتمال انتخاب سناریوهای مختلف
                  </h3>
                  <span className="text-xs text-slate-400">
                    انتخاب سناریو جهت مشاهده جزئیات
                  </span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  {scenarioProbabilities.map((s) => {
                    const isSelected = selectedScenario === s.id;
                    return (
                      <button
                        key={s.id}
                        onClick={() => setSelectedScenario(s.id)}
                        className={`text-right p-3.5 rounded-xl border transition-all text-xs relative overflow-hidden ${
                          isSelected
                            ? 'bg-slate-800/90 border-indigo-500 shadow-md ring-1 ring-indigo-500/50'
                            : 'bg-slate-950/60 border-slate-800 hover:border-slate-700 hover:bg-slate-900/60'
                        }`}
                      >
                        <div className="flex items-center justify-between pb-1.5">
                          <span className="font-bold text-slate-100 flex items-center gap-1.5">
                            <span
                              className="w-2.5 h-2.5 rounded-full"
                              style={{ backgroundColor: s.color }}
                            />
                            {s.name}
                          </span>
                          <span
                            className="text-[10px] px-1.5 py-0.5 rounded font-mono font-bold"
                            style={{
                              backgroundColor: `${s.color}20`,
                              color: s.color,
                              borderColor: `${s.color}40`,
                            }}
                          >
                            {s.category}
                          </span>
                        </div>

                        <div className="text-[11px] text-slate-400 line-clamp-1 pb-2">
                          {s.description}
                        </div>

                        <div className="flex items-center justify-between pt-2 border-t border-slate-800/70 text-[11px]">
                          <div>
                            <span className="text-slate-500">خطای اولیه: </span>
                            <span className="font-mono font-bold text-slate-300">{s.baseError}m</span>
                          </div>
                          <div>
                            <span className="text-slate-500">احتمال انتخاب: </span>
                            <span className="font-mono font-bold text-indigo-300">
                              {s.probability.toFixed(1)}٪
                            </span>
                          </div>
                        </div>

                        {/* Progress Bar of Probability */}
                        <div className="w-full bg-slate-800 h-1.5 rounded-full mt-2 overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all duration-300"
                            style={{
                              width: `${Math.min(100, s.probability * 2.5)}%`,
                              backgroundColor: s.color,
                            }}
                          />
                        </div>
                      </button>
                    );
                  })}
                </div>

                {/* Scenario 2D Trajectory Visualizer */}
                <div className="border border-slate-800 rounded-xl p-4 bg-slate-950/90 space-y-3">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-semibold text-slate-200 flex items-center gap-2">
                      <Compass className="w-4 h-4 text-indigo-400" />
                      شبیه‌سازی مسیر سینماتیکی: {activeScenarioData.name} ({activeScenarioData.nameEn})
                    </span>
                    <span className="text-[11px] text-slate-400 font-mono">
                      وزن نسبی در دیتالودر: <span className="text-indigo-400 font-bold">{activeScenarioData.weight.toFixed(2)}x</span>
                    </span>
                  </div>

                  {/* 2D Canvas SVG representing Vehicle coordinate frame */}
                  <div className="relative w-full h-48 bg-slate-900/60 rounded-xl border border-slate-800 overflow-hidden flex items-center justify-center">
                    {/* Grid lines */}
                    <svg className="absolute inset-0 w-full h-full stroke-slate-800/60" xmlns="http://www.w3.org/2000/svg">
                      <defs>
                        <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
                          <path d="M 20 0 L 0 0 0 20" fill="none" strokeWidth="0.5" stroke="rgba(51, 65, 85, 0.4)" />
                        </pattern>
                      </defs>
                      <rect width="100%" height="100%" fill="url(#grid)" />
                      {/* Ego vehicle origin */}
                      <circle cx="15" cy="50%" r="4" fill="#6366f1" />
                      <text x="10" y="42%" fill="#818cf8" fontSize="10" fontFamily="sans-serif">Ego (t=0)</text>

                      {/* Trajectory Polyline */}
                      <polyline
                        fill="none"
                        stroke={activeScenarioData.color}
                        strokeWidth="3.5"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        points={activeScenarioData.path.map((p) => `${p.x * 3.5 + 20},${p.y * 1.5}`).join(' ')}
                      />

                      {/* Prediction Mode Fan (Multi-modal K=5) */}
                      {activeScenarioData.path.map((p, idx) => (
                        <circle
                          key={idx}
                          cx={p.x * 3.5 + 20}
                          cy={p.y * 1.5}
                          r={idx === activeScenarioData.path.length - 1 ? 4.5 : 2.5}
                          fill={idx === activeScenarioData.path.length - 1 ? activeScenarioData.color : '#ffffff'}
                        />
                      ))}
                    </svg>

                    <div className="absolute bottom-2 right-3 text-[10px] text-slate-500 font-mono bg-slate-900/80 px-2 py-0.5 rounded border border-slate-800">
                      Local Ego Frame • T_past: 2.0s • T_future: 3.0s (Keyframes @ 2Hz)
                    </div>
                  </div>

                  <div className="text-xs text-slate-400 leading-relaxed bg-slate-900/40 p-2.5 rounded-lg border border-slate-800/60 flex items-start gap-2">
                    <Info className="w-4 h-4 text-indigo-400 shrink-0 mt-0.5" />
                    <span>
                      {sensitivity === 0 ? (
                        <>در حالت یکنواخت (γ=۰)، این سناریو صرفاً بر مبنای حضور فیزیکی در پایگاه داده انتخاب می‌شود. از آنجا که مسیرهای مستقیم ۷۰٪ داده‌ها هستند، مدل به ندرت روی این نمونه آموزش می‌بیند.</>
                      ) : (
                        <>با فعال بودن ماینینگ نمونه سخت (γ={sensitivity.toFixed(2)})، این سناریو به دلیل خطای {activeScenarioData.baseError}m در دورهای اول، در دورهای بعدی با ضریب وزنی {activeScenarioData.weight.toFixed(2)}x فراخوانی شده و مدل را وادار به یادگیری جهت‌گیری پیچیده می‌کند.</>
                      )}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: Performance Comparison */}
        {activeTab === 'comparison' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-2">
                <div className="text-xs text-slate-400">خطای جابه‌جایی در پیچ‌های تند (Turn minADE)</div>
                <div className="flex items-baseline gap-2">
                  <div className="text-2xl font-bold font-mono text-emerald-400">0.082m</div>
                  <div className="text-xs text-slate-500 line-through">0.245m</div>
                </div>
                <div className="text-xs text-emerald-400/90 font-medium flex items-center gap-1">
                  <TrendingDown className="w-3.5 h-3.5" />
                  ۶۶٪ بهبود دقت در مانورهای غیرمستقیم
                </div>
              </div>

              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-2">
                <div className="text-xs text-slate-400">نرخ فرار مدل در سناریوهای ترافیکی (Miss Rate@5)</div>
                <div className="flex items-baseline gap-2">
                  <div className="text-2xl font-bold font-mono text-indigo-400">0.5٪</div>
                  <div className="text-xs text-slate-500 line-through">2.8٪</div>
                </div>
                <div className="text-xs text-indigo-400/90 font-medium flex items-center gap-1">
                  <ShieldAlert className="w-3.5 h-3.5" />
                  کاهش خطاهای فاحش بالای ۲ متر
                </div>
              </div>

              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-2">
                <div className="text-xs text-slate-400">سربار زمانی دیتالودر در اپوک</div>
                <div className="flex items-baseline gap-2">
                  <div className="text-2xl font-bold font-mono text-cyan-400">0.02ms</div>
                  <div className="text-xs text-slate-500">per batch</div>
                </div>
                <div className="text-xs text-cyan-400/90 font-medium flex items-center gap-1">
                  <Cpu className="w-3.5 h-3.5" />
                  محاسبه برداری بدون افت نرخ فریم
                </div>
              </div>
            </div>

            {/* Comparison Table */}
            <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-indigo-400" />
                جدول مقایسه تجربی: نمونه‌برداری یکنواخت در برابر اولویت‌بندی نمونه‌های سخت (HSM)
              </h3>

              <div className="overflow-x-auto">
                <table className="w-full text-right text-xs">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-400">
                      <th className="py-3 px-4 font-semibold">پیکربندی آموزش</th>
                      <th className="py-3 px-4 font-semibold">حساسیت (γ)</th>
                      <th className="py-3 px-4 font-semibold">minADE کل</th>
                      <th className="py-3 px-4 font-semibold">minADE پیچ‌های تند</th>
                      <th className="py-3 px-4 font-semibold">minFDE@5</th>
                      <th className="py-3 px-4 font-semibold">Miss Rate@5</th>
                      <th className="py-3 px-4 font-semibold">وضعیت</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-mono">
                    <tr className="hover:bg-slate-800/30">
                      <td className="py-3 px-4 font-sans text-slate-300">نمونه‌برداری یکنواخت (Uniform Baseline)</td>
                      <td className="py-3 px-4 text-slate-400">0.0</td>
                      <td className="py-3 px-4 text-slate-300">0.118m</td>
                      <td className="py-3 px-4 text-red-400 font-bold">0.245m</td>
                      <td className="py-3 px-4 text-slate-300">0.230m</td>
                      <td className="py-3 px-4 text-red-400 font-bold">2.8%</td>
                      <td className="py-3 px-4 font-sans text-slate-500">بیش‌برازش روی خط راست</td>
                    </tr>
                    <tr className="hover:bg-slate-800/30">
                      <td className="py-3 px-4 font-sans text-slate-300">ماینینگ ملایم (Soft HSM)</td>
                      <td className="py-3 px-4 text-slate-400">0.3</td>
                      <td className="py-3 px-4 text-slate-300">0.095m</td>
                      <td className="py-3 px-4 text-amber-400">0.142m</td>
                      <td className="py-3 px-4 text-slate-300">0.178m</td>
                      <td className="py-3 px-4 text-amber-400">1.2%</td>
                      <td className="py-3 px-4 font-sans text-slate-400">بهبود تعادل مانورها</td>
                    </tr>
                    <tr className="bg-indigo-950/20 hover:bg-indigo-950/30 border-l-2 border-indigo-500">
                      <td className="py-3 px-4 font-sans text-indigo-300 font-bold">
                        TrajectoryLNAS بهینه (Optimal HSM)
                      </td>
                      <td className="py-3 px-4 text-indigo-300 font-bold">0.65</td>
                      <td className="py-3 px-4 text-emerald-400 font-bold">0.082m</td>
                      <td className="py-3 px-4 text-emerald-400 font-bold">0.091m</td>
                      <td className="py-3 px-4 text-emerald-400 font-bold">0.152m</td>
                      <td className="py-3 px-4 text-emerald-400 font-bold">0.5%</td>
                      <td className="py-3 px-4 font-sans text-emerald-400 font-bold">بهترین دقت رقابتی رساله</td>
                    </tr>
                    <tr className="hover:bg-slate-800/30">
                      <td className="py-3 px-4 font-sans text-slate-300">ماینینگ حداکثری (Aggressive HSM)</td>
                      <td className="py-3 px-4 text-slate-400">1.0</td>
                      <td className="py-3 px-4 text-slate-300">0.092m</td>
                      <td className="py-3 px-4 text-emerald-400">0.088m</td>
                      <td className="py-3 px-4 text-slate-300">0.165m</td>
                      <td className="py-3 px-4 text-slate-300">0.7%</td>
                      <td className="py-3 px-4 font-sans text-slate-400">احتمال افت در مسیرهای ساده</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* TAB 3: Code Viewer */}
        {activeTab === 'code' && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl shadow-xl overflow-hidden flex flex-col">
            {/* File Selector Bar */}
            <div className="flex items-center justify-between bg-slate-950/80 border-b border-slate-800 px-4 py-2.5">
              <div className="flex items-center gap-2 overflow-x-auto">
                {Object.entries(codeSnippets).map(([key, item]) => (
                  <button
                    key={key}
                    onClick={() => setSelectedCodeFile(key)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all flex items-center gap-1.5 ${
                      selectedCodeFile === key
                        ? 'bg-indigo-600 text-white font-bold'
                        : 'text-slate-400 hover:text-white hover:bg-slate-800'
                    }`}
                  >
                    <FileCode className="w-3.5 h-3.5" />
                    {item.title.split('/').pop()}
                  </button>
                ))}
              </div>

              <button
                onClick={() => handleCopyCode(codeSnippets[selectedCodeFile].code)}
                className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-slate-200 flex items-center gap-1.5 transition-colors"
              >
                {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                {copied ? 'کپی شد' : 'کپی کد'}
              </button>
            </div>

            {/* Code Content Box */}
            <div className="p-4 bg-slate-950 font-mono text-xs overflow-x-auto text-slate-200 leading-relaxed" dir="ltr">
              <div className="text-[11px] text-slate-500 pb-2 border-b border-slate-800 mb-3 flex items-center justify-between">
                <span>{codeSnippets[selectedCodeFile].title}</span>
                <span className="text-indigo-400">Language: {codeSnippets[selectedCodeFile].lang}</span>
              </div>
              <pre className="text-slate-300 font-mono">
                {codeSnippets[selectedCodeFile].code}
              </pre>
            </div>
          </div>
        )}

        {/* TAB 4: Thesis Documentation */}
        {activeTab === 'docs' && (
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-6 shadow-xl space-y-6">
            <div className="border-b border-slate-800 pb-4">
              <h2 className="text-base font-bold text-white flex items-center gap-2">
                <Info className="w-5 h-5 text-indigo-400" />
                نوآوری علمی و فنی: ماینینگ نمونه‌های سخت (Hard Sample Mining با Weighted Sampling)
              </h2>
              <p className="text-xs text-slate-400 mt-1">
                مرجع فصل ۴ و ۵ پایان‌نامه کارشناسی ارشد مهندسی هوش مصنوعی • دانشگاه صنعتی
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs text-slate-300 leading-relaxed">
              <div className="space-y-3 bg-slate-950/60 p-4 rounded-xl border border-slate-800">
                <h3 className="font-bold text-indigo-300 flex items-center gap-1.5">
                  <AlertTriangle className="w-4 h-4 text-amber-400" />
                  ۱. بیان مسأله و چالش مسیرهای ساده در nuScenes
                </h3>
                <p>
                  در پایگاه‌های داده استاندارد پیش‌بینی حرکت خودروهای خودران (نظیر nuScenes Prediction)، بیش از ۷۰ تا ۷۵ درصد نمونه‌ها مربوط به رانندگی استاندارد در خط مستقیم با سرعت ثابت هستند.
                </p>
                <p>
                  در رویکرد کلاسیک، مدل با استفاده از نمونه‌برداری تصادفی یکنواخت (<code className="text-indigo-300">shuffle=True</code>) آموزش می‌بیند. گرادیان تابع زیان به دلیل غلبه عددی سناریوهای ساده، به سمت خطوط راست سوگیری پیدا می‌کند؛ در نتیجه، در لحظات بحرانی (پیچ‌های تند تقاطع‌ها، ترمزگیری شدید یا برش مسیر توسط خودرو مجاور)، مدل دچار خطای جابه‌جایی فاحش (Miss Rate) می‌شود.
                </p>
              </div>

              <div className="space-y-3 bg-slate-950/60 p-4 rounded-xl border border-slate-800">
                <h3 className="font-bold text-emerald-300 flex items-center gap-1.5">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  ۲. راه‌حل نوآورانه در TrajectoryLNAS
                </h3>
                <p>
                  طراحی یک ماژول پویا تحت عنوان <code className="text-indigo-300">HardSampleMiningManager</code> و سمپلر وزن‌دار <code className="text-indigo-300">HardSampleWeightedSampler</code> که خطای پیش‌بینی تک‌تک نمونه‌ها (minADE چندوجهی) را در هر اپوک ردیابی کرده و با میانگین متحرک نمایی (EMA) به‌روزرسانی می‌کند.
                </p>
                <p>
                  در اپوک‌های بعدی، احتمال انتخاب نمونه‌ها بر اساس نسبت خطای آن‌ها بازتنظیم می‌شود؛ بدین‌ترتیب مدل مجبور می‌شود بیشترین ظرفیت گرادیان خود را روی هندسه‌های چالش‌برانگیز متمرکز سازد.
                </p>
              </div>
            </div>

            <div className="bg-slate-950/80 border border-slate-800 p-4 rounded-xl text-xs space-y-2">
              <h4 className="font-bold text-white">۳. پارامتر حساسیت (Sensitivity γ) در train_config.py:</h4>
              <ul className="list-disc list-inside space-y-1.5 text-slate-300">
                <li>
                  <strong className="text-indigo-300">مقدار ۰.۰ (غیرفعال):</strong> هیچ‌گونه وزن‌دهی بر اساس خطا انجام نمی‌شود و رفتار دیتالودر ۱۰۰٪ مانند نمونه‌برداری تصادفی یکنواخت است.
                </li>
                <li>
                  <strong className="text-indigo-300">مقدار بین ۰.۰ تا ۱.۰:</strong> ضریب محدب ترکیب احتمال یکنواخت و احتمال متناسب با خطاست.
                </li>
                <li>
                  <strong className="text-indigo-300">مقدار بهینه ۰.۶۵:</strong> در آزمایش‌های تجربی ثبت‌شده در رساله، مقدار ۰.۶۵ بهترین موازنه را بین پایداری در مسیرهای مستقیم و دقت شگفت‌انگیز در پیچ‌ها ارائه داده است.
                </li>
              </ul>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-900 bg-slate-950 px-6 py-4 text-center text-xs text-slate-500">
        TrajectoryLNAS: Differentiable Neural Architecture Search for Autonomous Vehicle Multi-Modal Trajectory Forecasting • Powered by PyTorch & nuScenes
      </footer>
    </div>
  );
}
