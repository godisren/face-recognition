@echo Off

SET current_path=%~dp0
echo %current_path%

call .\venv\Scripts\activate

set FLASK_APP=face
set FLASK_ENV=development
set SETTINGS=%current_path%\settings.cfg