# تمرین Git برای پروژه DebCo

## 1. ساخت مخزن محلی

```bash
git init
git status
git add .
git commit -m "chore: bootstrap clean debco research engine"
```

## 2. ساخت branch برای هر مرحله

برای مرحله دریافت داده:

```bash
git checkout -b feature/data-fetch-dxy
```

بعد از تغییرات:

```bash
git add .
git commit -m "feat: add mt5 data fetch and dxy builder"
git checkout main
git merge feature/data-fetch-dxy
```

## 3. اتصال به GitHub

اول در GitHub یک repo خالی بساز، مثلا:

```text
debco-research-engine
```

بعد:

```bash
git remote add origin https://github.com/YOUR_USERNAME/debco-research-engine.git
git branch -M main
git push -u origin main
```

## 4. نسخه گذاری

بعد از هر milestone:

```bash
git tag v0.1.0
git push origin v0.1.0
```

## قانون پروژه

هیچ فایل zip پراکنده مبنای توسعه نیست. هر کد قابل استفاده باید داخل Git باشد.
