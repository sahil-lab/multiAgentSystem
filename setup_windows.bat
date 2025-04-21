@echo off
echo Setting up Multi-Agent AI System environment...

:: Create necessary directories
mkdir models 2>nul
mkdir chroma_db 2>nul

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo Python found, installing dependencies...
    
    :: Install dependencies using pip
    python -m pip install -r requirements.txt
    
    if %errorlevel% equ 0 (
        echo Dependencies installed successfully!
    ) else (
        echo Error installing dependencies. Please check requirements.txt and try again.
        exit /b 1
    )
) else (
    echo Python not found. Please install Python 3.8 or higher and try again.
    exit /b 1
)

echo Do you want to download model files now? (y/n)
set /p download_choice=

if /i "%download_choice%"=="y" (
    echo Which model would you like to download?
    echo 1. Llama 2 7B Chat (4-bit quantized, ~4GB)
    echo 2. Mistral 7B Instruct v0.2 (4-bit quantized, ~4GB)
    echo 3. NeuralChat 7B v3.1 (4-bit quantized, ~4GB)
    echo 4. Phi-2 (4-bit quantized, ~2GB)
    echo 5. All of the above
    echo 0. None/Skip
    
    set /p model_choice=
    
    if "%model_choice%"=="1" (
        call :download_model "llama-2-7b-chat.Q4_K_M.gguf" "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"
    ) else if "%model_choice%"=="2" (
        call :download_model "mistral-7b-instruct-v0.2.Q4_K_M.gguf" "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    ) else if "%model_choice%"=="3" (
        call :download_model "neural-chat-7b-v3-1.Q4_K_M.gguf" "https://huggingface.co/TheBloke/neural-chat-7B-v3-1-GGUF/resolve/main/neural-chat-7b-v3-1.Q4_K_M.gguf"
    ) else if "%model_choice%"=="4" (
        call :download_model "phi-2.Q4_K_M.gguf" "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf"
    ) else if "%model_choice%"=="5" (
        call :download_model "llama-2-7b-chat.Q4_K_M.gguf" "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"
        call :download_model "mistral-7b-instruct-v0.2.Q4_K_M.gguf" "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        call :download_model "neural-chat-7b-v3-1.Q4_K_M.gguf" "https://huggingface.co/TheBloke/neural-chat-7B-v3-1-GGUF/resolve/main/neural-chat-7b-v3-1.Q4_K_M.gguf"
        call :download_model "phi-2.Q4_K_M.gguf" "https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf"
    ) else (
        echo Skipping model download. You'll need to place model files in the 'models' directory manually.
    )
) else (
    echo Skipping model download. You'll need to place model files in the 'models' directory manually.
)

echo.
echo Setup complete! To start the system, run: python app.py
echo Then open your web browser to http://localhost:7860
pause
exit /b 0

:download_model
set MODEL_NAME=%~1
set MODEL_URL=%~2

if not exist "models\%MODEL_NAME%" (
    echo Downloading %MODEL_NAME%...
    
    powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%MODEL_URL%' -OutFile 'models\%MODEL_NAME%'}"
    
    if %errorlevel% equ 0 (
        echo %MODEL_NAME% downloaded successfully!
    ) else (
        echo Error downloading %MODEL_NAME%.
        exit /b 1
    )
) else (
    echo %MODEL_NAME% already exists, skipping download.
)

exit /b 0 