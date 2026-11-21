# TrajectoryLNAS: Trajectory Prediction Architecture Optimization via Lightweight Neural Architecture Search
**پروژه پایان‌نامه کارشناسی ارشد مهندسی کامپیوتر - هوش مصنوعی و رباتیک**  
دانشجو: علیرضا غفوری (شماره دانشجویی: 4031631209)  
استاد راهنما: دکتر زینب قاسمی نراقی | استاد مشاور: دکتر زینب عابدینی  
شماره ثبت پژوهش: ۲۸۲۴۵۲۵۹ | شماره ثبت ایرانداک: ۱۹۸۷۵۶۸

### مراجع و مستندات رسمی پروژه:
- **معماری سیستم، نوآوری‌های علمی و راهنمای پیاده‌سازی:** [docs/SYSTEM_ARCHITECTURE_AND_INNOVATIONS.md](docs/SYSTEM_ARCHITECTURE_AND_INNOVATIONS.md)
- **ممیزی بنچمارک‌های SOTA، نتایج تجربی و اعتبارسنجی:** [docs/BENCHMARKS_AND_VERIFICATION_REPORT.md](docs/BENCHMARKS_AND_VERIFICATION_REPORT.md)

---

## ساختار پوشه‌ها و فایل‌ها (Codebase Structure)
```
/
├── main.py                     # نقطه ورود اصلی اجرای پایتون (مدیریت خط‌لوله جستجو، آموزش و ارزیابی)
├── configs/                    # پوشه پیکربندی پارامترها و تنظیمات مدل، داده و سخت‌افزار
│   ├── model_config.py
│   ├── data_config.py
│   ├── search_config.py
│   ├── train_config.py
│   └── hardware_config.py
├── data/                       # مجموعه‌های داده و خط‌لوله پیش‌پردازش
│   ├── nuscenes_dataset.py     # لودر اختصاصی nuScenes و Synthetic Dataset
│   ├── preprocessing.py        # پیش‌پردازش و استخراج لایه‌های داده (جایگزین prepare_data.py)
│   └── data_validation.py      # اعتبارسنجی کیفیت داده‌ها و تست‌های پذیرش (ادغام test_acceptance و data_summary)
├── core/                       # هسته الگوریتمی و جستجوی معماری
│   ├── darts_algorithm.py      # الگوریتم جستجوی مشتق‌پذیر DARTS (دارای کلاس DARTSAlgorithm)
│   └── progressive_search.py   # موتور اجرای جستجوی تدریجی (جایگزین search.py سابق)
├── models/                     # معماری شبکه‌ها و سوپرنت
│   ├── supernet_operators.py   # عملگرهای سوپرنت و MixedOps
│   ├── search_supernet.py      # شبکه فراگیر (Super-Net) و پارامترهای معماری ArchParams
│   └── final_model.py          # معماری استخراج‌شده گسسته (FinalTrajectoryModel)
├── train/                      # کلاس‌ها و اسکریپت‌های آموزش
│   ├── base_trainer.py         # کلاس پایه ترینر
│   ├── final_trainer.py        # ترینر مدل نهایی
│   └── train_final.py          # اسکریپت اجرای آموزش نهایی با Early Stopping
├── evaluations/                # ارزیابی، معیارهای عملکرد، پروکسی‌ها و پروفایلر سخت‌افزاری
│   ├── evaluator.py            # موتور اصلی ارزیابی مدلی روی بنچمارک‌ها
│   ├── metrics.py              # محاسبات تحلیلی ADE, FDE, MissRate، پروکسی‌ها و پروفایلر سخت‌افزاری
│   └── visualization.py        # رسم نمودارهای علمی، مقایسه‌ای و پلات‌های افق پیش‌بینی
├── utils/                      # ابزارهای عمومی و سیستم لاگ
│   ├── __init__.py             # ماژول یکپارچه توابع عمومی و ری‌اسپورت‌ها
│   ├── seed.py                 # تنظیم سید تصادفی
│   ├── dataloaders.py          # لودرهای عمومی داده
│   └── logger.py               # سیستم دوگانه ثبت لاگ (DualLogger)
├── experiments/                # آزمایش‌ها، ابلیشن و بنچمارک‌ها
│   ├── ablation.py
│   ├── baselines.py
│   └── benchmark.py
├── checkpoints/                # پوشه ذخیره وزن‌ها و چک‌پوینت‌های مدل (مثلاً best_trajectory_lnas.pt)
└── reports/                    # پوشه ذخیره لاگ‌های اجرا، نتایج JSON و نمودارهای ارزیابی (جایگزین report/)
```

---

## آدرس‌ها و راهنمای دیتاست‌ها (Datasets & Directory Paths)

### ۱. دیتاست nuScenes Motion Prediction & Synthetic
* **لینک دریافت رسمی**: [https://www.nuscenes.org/download](https://www.nuscenes.org/download)
* **ابعاد داده**: فریم‌های کلیدی nuScenes با فرکانس استاندارد ۲ هرتز (Keyframes @ 2 Hz؛ گذشته ۴ گام و آینده ۱۲ گام) نمونه‌برداری می‌شوند. همچنین پشتیبانی کامل از Synthetic Dataset برای تست سریع وجود دارد.

---

## نحوه اجرا (How to Run)

### ۱. نصب وابستگی‌ها:
```bash
pip install -r requirements.txt
```

### ۲. اجرای خط‌لوله کامل ۳ مرحله‌ای:
```bash
python main.py --mode search --dataset synthetic
```

### ۳. اجرای اسکریپت‌های تخصصی ماژولار:
* **جستجوی معماری (DARTS):**
  ```bash
  python core/progressive_search.py --config configs/search_config.py
  ```
* **آموزش نهایی:**
  ```bash
  python train/train_final.py --epochs 50
  ```
* **ارزیابی و پروفایلینگ:**
  ```bash
  python evaluations/evaluator.py
  ```

---

## خروجی‌ها و گزارش‌های تولیدشده (Generated Artifacts)
پس از اجرای موفق، کلیه نتایج به صورت خودکار در مسیرهای زیر ذخیره می‌شوند:
1. `checkpoints/best_trajectory_lnas.pt`: اوزان بهینه استخراج‌شده از بهترین اپوک مدل نهایی.
2. `reports/search_execution.log`: لاگ متنی کامل اجرای خط‌لوله.
3. `reports/real_experiment_results.json`: فایل ساختاریافته شامل معیارهای ZCP، آلفاها، تاریخچه آموزش و پروفایل سخت‌افزاری.
4. `reports/horizon_error_curve.png`: نمودار خطای جابه‌جایی در افق‌های ۱، ۲ و ۳ ثانیه.
5. `reports/sota_benchmark_comparison.png`: نمودار میله‌ای مقایسه عملکرد با مدل‌های مطرح دنیا.

## قرارداد معیارهای ارزیابی و هزینه

- معیارهای `min_ade_k`، `min_fde_k` و `miss_rate_k` ورودی اختیاری `mode_logits` را برای انتخاب محتمل‌ترین K مسیر می‌پذیرند. در صورت حذف آن، ترتیب مسیرهای ورودی باید از قبل بر اساس احتمال نزولی باشد.
- `miss_rate_k` نسبت نمونه‌هایی را برمی‌گرداند که در تمام مسیرهای منتخب، بیشترین خطای زمانی حداقل برابر آستانه است؛ ضرب در ۱۰۰ هنگام گزارش انجام می‌شود. این معیار با نسخهٔ قبلی مبتنی بر نقطهٔ پایانی قابل مقایسهٔ مستقیم نیست و گزارش‌های قبلی باید دوباره ارزیابی شوند.
- برای برخورد، ماسک `[B, N]` اعتبار همسایه و ماسک `[B, T, N]` اعتبار زمانی را مشخص می‌کند. بدون ماسک، تمام همسایه‌ها و زمان‌های ورودی معتبر محسوب می‌شوند؛ مختصات صفر نشانهٔ padding نیست.
- تابع `simple_complexity_proxy(alphas, all_costs)` به هزینهٔ هر عملگر نیاز دارد و هزینهٔ موردانتظار را بر حسب GFLOPs برمی‌گرداند. پروفایلر برای هزینهٔ محاسباتی نامعلوم مقدار `None` گزارش می‌کند.
- FLOPs مقدار تحلیلی تقریبی با قرارداد دو عملیات برای هر ضرب‌وجمع است؛ هزینهٔ دیکودر GRU شامل تبدیل‌های ورودی و حالت پنهان است. Cosine Annealing فقط در فاز آموزش نهایی فعال است.

آزمون‌های رگرسیون محاسباتی بدون دسترسی به دیتاست اجرا می‌شوند:

```bash
python -m unittest discover -s tests -p test_mathematical_correctness.py -v
```
