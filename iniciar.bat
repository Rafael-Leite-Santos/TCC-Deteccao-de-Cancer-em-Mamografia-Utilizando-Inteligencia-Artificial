@echo off
rem ================================================================
rem  iniciar.bat - liga a interface web do TCC (duplo clique)
rem
rem  1. ativa o ambiente conda que tem o mmdet
rem  2. instala as bibliotecas da interface que faltarem
rem  3. roda o verificar.py
rem  4. liga o servidor e abre o navegador
rem
rem  As mensagens estao sem acento de proposito: o cmd do Windows
rem  costuma mostrar acentos errado.
rem ================================================================
chcp 65001 >nul
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
title Deteccao de Massas - servidor

echo.
echo ================================================================
echo   Deteccao de Massas em Mamografia - iniciando
echo ================================================================
echo.

rem ---------------------------------------------------------------
rem  1. Achar o conda
rem ---------------------------------------------------------------
echo [1/4] Procurando o Anaconda/Miniconda...
set "CONDA_ROOT="
for /f "delims=" %%i in ('conda info --base 2^>nul') do set "CONDA_ROOT=%%i"
if not defined CONDA_ROOT for /f "delims=" %%i in ('mamba info --base 2^>nul') do set "CONDA_ROOT=%%i"
for %%d in ("%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%USERPROFILE%\miniforge3" "%USERPROFILE%\mambaforge" "%LOCALAPPDATA%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%ProgramData%\miniconda3" "%ProgramData%\anaconda3" "C:\miniconda3" "C:\anaconda3") do (
    if not defined CONDA_ROOT if exist "%%~d\Scripts\activate.bat" set "CONDA_ROOT=%%~d"
)
if not defined CONDA_ROOT goto sem_conda
if not exist "%CONDA_ROOT%\Scripts\activate.bat" goto sem_conda
echo       Conda em: %CONDA_ROOT%

rem ---------------------------------------------------------------
rem  2. Escolher o ambiente: o salvo antes, ou o unico com mmdet
rem ---------------------------------------------------------------
set "AMBIENTE="
if exist ambiente_local.txt set /p AMBIENTE=<ambiente_local.txt
if defined AMBIENTE goto ativar

set "QTD=0"
set "ACHADO="
echo       Ambientes com o mmdet instalado:
for /d %%e in ("%CONDA_ROOT%\envs\*" "%USERPROFILE%\.conda\envs\*") do (
    if exist "%%~e\Lib\site-packages\mmdet*" call :achou "%%~e"
)
if exist "%CONDA_ROOT%\Lib\site-packages\mmdet*" call :achou "%CONDA_ROOT%"
if "%QTD%"=="0" echo         nenhum encontrado automaticamente
if not "%QTD%"=="1" goto perguntar
set "AMBIENTE=%ACHADO%"
goto ativar

:perguntar
echo.
echo   Digite o NOME do ambiente conda que voce usou no treino
echo   - o mesmo do comando: conda activate NOME
echo   A resposta fica salva no arquivo ambiente_local.txt.
echo.
set "AMBIENTE="
set /p "AMBIENTE=  Nome do ambiente: "
if not defined AMBIENTE goto perguntar
>ambiente_local.txt echo %AMBIENTE%

:ativar
echo       Ativando o ambiente: %AMBIENTE%
call "%CONDA_ROOT%\Scripts\activate.bat" "%AMBIENTE%"
if errorlevel 1 goto ativar_falhou
goto dependencias

:ativar_falhou
echo.
echo   [X] Nao consegui ativar o ambiente "%AMBIENTE%".
if exist ambiente_local.txt del ambiente_local.txt
goto perguntar

:sem_conda
echo       Nao encontrei o conda neste computador.
where python >nul 2>nul || goto sem_python
echo       Vou usar o Python que estiver no PATH.
goto dependencias

:sem_python
echo.
echo   [X] Nao encontrei nem o conda nem o Python.
echo       Abra o "Anaconda Prompt", entre nesta pasta e rode:  python verificar.py
echo.
pause
exit /b 1

rem ---------------------------------------------------------------
rem  3. Bibliotecas da interface + verificacao
rem ---------------------------------------------------------------
:dependencias
echo.
echo [2/4] Conferindo as bibliotecas da interface web...
python -m pip install --disable-pip-version-check -q -r requirements-web.txt
if errorlevel 1 echo       Nao consegui instalar agora - sem internet? O verificador mostra o que falta.

echo.
echo [3/4] Verificando se esta tudo pronto...
python verificar.py
set "RESULTADO=%errorlevel%"
if "%RESULTADO%"=="2" goto nao_inicia
if "%RESULTADO%"=="1" goto aviso_demo
goto ligar

:nao_inicia
echo   O servidor nao pode ser ligado. Corrija os itens com X acima.
echo.
pause
exit /b 1

:aviso_demo
echo   ATENCAO: o servidor vai abrir em MODO DEMONSTRACAO.
echo   As caixas na tela serao FALSAS. Nao use assim na apresentacao.
echo.
echo   Para continuar mesmo assim, aperte qualquer tecla.
echo   Para corrigir antes, feche esta janela.
pause >nul

rem ---------------------------------------------------------------
rem  4. Ligar o servidor e abrir o navegador
rem ---------------------------------------------------------------
:ligar
powershell -NoProfile -Command "try { (New-Object Net.Sockets.TcpClient('127.0.0.1', 8000)).Close(); exit 0 } catch { exit 1 }" >nul 2>nul
if not errorlevel 1 goto ja_ligado

rem Abre o navegador sozinho assim que o servidor responder (o modelo pode demorar a carregar)
start "" /min powershell -NoProfile -WindowStyle Hidden -Command "for ($i = 0; $i -lt 300; $i++) { try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 http://localhost:8000/status | Out-Null; Start-Process 'http://localhost:8000'; break } catch { Start-Sleep -Seconds 1 } }"

echo.
echo [4/4] Ligando o servidor...
echo ================================================================
echo   O navegador abre sozinho quando o modelo terminar de carregar.
echo   Endereco: http://localhost:8000
echo.
echo   FECHE ESTA JANELA PARA DESLIGAR O SERVIDOR.
echo ================================================================
echo.
python -m uvicorn main:app --port 8000
echo.
echo   O servidor parou. Veja a mensagem acima.
pause
exit /b 0

:ja_ligado
echo.
echo   O servidor ja esta ligado - talvez em outra janela.
echo   Abrindo o navegador em http://localhost:8000
start "" http://localhost:8000
echo.
pause
exit /b 0

rem ---------------------------------------------------------------
:achou
set /a QTD+=1
set "ACHADO=%~1"
echo         - %~1
exit /b 0
