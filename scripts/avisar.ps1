<#
.SYNOPSIS
Windows notification of the paper routines: the 12h watchdog (vigia.ps1) and the daily upkeep (manutencao.py).

.DESCRIPTION
Title and body come as parameters (from PowerShell) or from the AVISO_TITULO and AVISO_CORPO environment variables
(from Python: Windows PowerShell 5.1 mangles double quotes inside native command-line arguments). Shown under the
PowerShell app id, like the watchdog always did.

.EXAMPLE
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\avisar.ps1 -Titulo 'teste' -Corpo 'corpo' -SoMostrar
#>
param(
    [string]$Titulo = $env:AVISO_TITULO,
    [string]$Corpo = $env:AVISO_CORPO,
    [switch]$SoMostrar      # writes "title | body" to the output instead of notifying (tests)
)

$ErrorActionPreference = 'Stop'
$AppDoPowerShell = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe'

if ($SoMostrar) {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    "$Titulo | $Corpo"
    exit 0
}
$null = [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime]
$null = [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime]
$xml = New-Object Windows.Data.Xml.Dom.XmlDocument
$xml.LoadXml('<toast><visual><binding template="ToastGeneric"><text/><text/></binding></visual></toast>')
$textos = $xml.GetElementsByTagName('text')
$null = $textos.Item(0).AppendChild($xml.CreateTextNode($Titulo))
$null = $textos.Item(1).AppendChild($xml.CreateTextNode($Corpo))
$aviso = [Windows.UI.Notifications.ToastNotification]::new($xml)
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($AppDoPowerShell).Show($aviso)
