; NSIS Installer Script for Apah Sovereign AI Desktop Engine
; Generates Windows setup installer wrapping PyInstaller executable.

!define APP_NAME "Apah Sovereign AI Engine"
!define COMP_NAME "Apah Project"
!define VERSION "0.5.0"
!define EXE_NAME "apah.exe"
!define REG_KEY "HKCU"
!define REG_SUBKEY "Software\Microsoft\Windows\CurrentVersion\Run"

Name "${APP_NAME}"
OutFile "ApahSetup-${VERSION}.exe"
InstallDir "$PROGRAMFILES64\Apah"
InstallDirRegKey HKCU "Software\Apah" "InstallDir"
RequestExecutionLevel admin

; UI Pages
Page directory
Page components
Page instfiles

UninstPage uninstConfirm
UninstPage instfiles

Section "Apah Application (Required)" SecCore
  SectionIn RO
  SetOutPath "$INSTDIR"
  
  File "..\..\dist\apah.exe"
  
  ; Write uninstaller
  WriteUninstaller "$INSTDIR\uninstall.exe"
  
  ; Start Menu Shortcuts
  CreateDirectory "$SMPROGRAMS\Apah"
  CreateShortcut "$SMPROGRAMS\Apah\Apah Dashboard.lnk" "$INSTDIR\apah.exe"
  CreateShortcut "$SMPROGRAMS\Apah\Uninstall Apah.lnk" "$INSTDIR\uninstall.exe"
  
  ; Add/Remove Programs Registry
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\Apah" "DisplayName" "${APP_NAME}"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\Apah" "UninstallString" "$INSTDIR\uninstall.exe"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\Apah" "DisplayVersion" "${VERSION}"
SectionEnd

Section /o "Create Desktop Shortcut" SecDesktopShortcut
  CreateShortcut "$DESKTOP\Apah Dashboard.lnk" "$INSTDIR\apah.exe"
SectionEnd

Section /o "Launch Apah on Windows Startup" SecStartupRegistry
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Run" "ApahEngine" "$INSTDIR\apah.exe"
SectionEnd

; Uninstaller Section
Section "Uninstall"
  Delete "$INSTDIR\apah.exe"
  Delete "$INSTDIR\uninstall.exe"
  Delete "$DESKTOP\Apah Dashboard.lnk"
  
  RMDir /r "$SMPROGRAMS\Apah"
  RMDir /r "$INSTDIR"
  
  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\Apah"
  DeleteRegValue HKCU "Software\Microsoft\Windows\CurrentVersion\Run" "ApahEngine"
SectionEnd
