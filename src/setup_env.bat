@echo off
REM ============================================================
REM Logo Recognition 环境配置脚本 (Windows)
REM Based on: "Scalable Logo Recognition using Proxies"
REM ============================================================

echo ============================================================
echo Logo Recognition 环境配置
echo ============================================================
echo.

REM 检查 conda 是否可用
where conda >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [错误] 未找到 conda，请先安装 Anaconda 或 Miniconda
    echo 下载地址: https://www.anaconda.com/download
    pause
    exit /b 1
)

echo [步骤 1/5] 创建 conda 环境: logo_recognition
echo.

REM 创建环境
conda create -n logo_recognition python=3.9 -y
if %ERRORLEVEL% neq 0 (
    echo [警告] 环境可能已存在，尝试继续...
)

echo.
echo [步骤 2/5] 激活环境
echo.
call conda activate logo_recognition

echo.
echo [步骤 3/5] 安装 PyTorch (CUDA 11.8)
echo 如果没有GPU，将自动使用CPU版本
echo.

REM 检测是否有 NVIDIA GPU
nvidia-smi >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo 检测到 NVIDIA GPU，安装 CUDA 版本...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else (
    echo 未检测到 GPU，安装 CPU 版本...
    pip install torch torchvision torchaudio
)

echo.
echo [步骤 4/5] 安装其他依赖
echo.
pip install numpy Pillow tqdm opencv-python scikit-learn matplotlib tensorboard PyYAML

echo.
echo [步骤 5/5] 验证安装
echo.
python -c "import torch; print(f'PyTorch 版本: {torch.__version__}'); print(f'CUDA 可用: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"无\"}')"

echo.
echo ============================================================
echo 环境配置完成!
echo ============================================================
echo.
echo 使用方法:
echo   1. 激活环境:    conda activate logo_recognition
echo   2. 准备数据:    python prepare_data.py
echo   3. 运行测试:    python demo.py
echo   4. 开始训练:    python train.py --help
echo.
pause
