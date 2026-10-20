<#
.SYNOPSIS
Vigia das 12h do paper trading: confere se a rotina do lago e a do paper rodaram hoje e deram certo, e avisa no
Windows quando não.

.DESCRIPTION
Lê o runs.csv do lago (%LOCALAPPDATA%\b3lake, formato do b3lake) e o do paper (%LOCALAPPDATA%\paper-trading,
formato do anotar da rotina) e o state.json do lago. Problemas: rotina sem linha hoje; última rodada de hoje do lago
"failed" ou "missing_overdue"; última de hoje do paper "falhou", "bloqueada" ou "lago rodando"; paper decidido num
fechamento mais velho que o último que o lago já tem. Tarefa "paper vigia", seg-sex 12h. Antes de conferir, espera
as rotinas que estiverem rodando: com o PC ligado depois do horário, as tarefas atrasadas rodam juntas.

.EXAMPLE
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\vigia.ps1 -SoMostrar
#>
param(
    [string]$Hoje = (Get-Date -Format 'yyyy-MM-dd'),
    [string]$PastaLago = (Join-Path $env:LOCALAPPDATA 'b3lake'),
    [string]$PastaPaper = (Join-Path $env:LOCALAPPDATA 'paper-trading'),
    [switch]$SoMostrar,     # escreve os problemas na saída em vez de avisar no Windows (teste e conferência à mão)
    [switch]$SemEspera      # não espera as rotinas que estiverem rodando (teste)
)

$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$RotinasVigiadas = @('b3lake daily', 'paper trading')
$EsperaMaxima = [TimeSpan]::FromMinutes(45)
$FalhasDoLago = @('failed', 'missing_overdue')
$FalhasDoPaper = @('falhou', 'bloqueada', 'lago rodando')
$AppDoPowerShell = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe'

function Esperar-Rotinas {
    # rodada atrasada (PC ligado ou acordado depois das 12h): as outras tarefas atrasadas começam junto com esta
    if ((Get-Date).TimeOfDay -gt [TimeSpan]'12:10:00') { Start-Sleep -Seconds 180 }
    $limite = (Get-Date) + $EsperaMaxima
    while ((Get-Date) -lt $limite) {
        $rodando = @(Get-ScheduledTask -TaskName $RotinasVigiadas -ErrorAction SilentlyContinue |
                     Where-Object State -eq 'Running')
        if ($rodando.Count -eq 0) { return }
        Start-Sleep -Seconds 30
    }
}

function Ler-Rodadas([string]$Arquivo) {
    if (-not (Test-Path -LiteralPath $Arquivo)) { return @() }
    return @(Import-Csv -LiteralPath $Arquivo -Encoding UTF8)
}

function Ler-UltimoDoLago([string]$Pasta) {
    $estado = Join-Path $Pasta 'state.json'
    if (-not (Test-Path -LiteralPath $estado)) { return $null }
    return (Get-Content -LiteralPath $estado -Raw -Encoding UTF8 | ConvertFrom-Json).last_done
}

function Problemas-DoDia([string]$Dia, [object[]]$DoLago, [object[]]$DoPaper, [string]$UltimoDoLago) {
    $problemas = New-Object System.Collections.Generic.List[string]
    $lagoHoje = @($DoLago | Where-Object { "$($_.started)".StartsWith($Dia) })
    if ($lagoHoje.Count -eq 0) {
        $problemas.Add('lago: nenhuma rodada hoje (tarefa "b3lake daily")')
    } elseif ($FalhasDoLago -contains $lagoHoje[-1].outcome) {
        $problemas.Add("lago: $($lagoHoje[-1].outcome) ($($lagoHoje[-1].detail))")
    }
    $paperHoje = @($DoPaper | Where-Object { "$($_.inicio)".StartsWith($Dia) })
    if ($paperHoje.Count -eq 0) {
        $problemas.Add('paper trading: nenhuma rodada hoje (tarefa "paper trading")')
    } elseif ($FalhasDoPaper -contains $paperHoje[-1].desfecho) {
        $problemas.Add("paper trading: $($paperHoje[-1].desfecho) ($($paperHoje[-1].nota))")
    }
    if (-not $UltimoDoLago) {
        $problemas.Add('lago: sem o state.json (último fechamento desconhecido)')
        return $problemas
    }
    $fechamentos = @($DoPaper | ForEach-Object { "$($_.fechamento)" } | Where-Object { $_ } | Sort-Object)
    $ultimoDoPaper = if ($fechamentos.Count -gt 0) { $fechamentos[-1] } else { $null }
    if (-not $ultimoDoPaper -or $ultimoDoPaper -lt $UltimoDoLago) {
        $problemas.Add("paper trading: último fechamento decidido $ultimoDoPaper, o lago já tem o de $UltimoDoLago")
    }
    return $problemas
}

function Avisar([string]$Titulo, [string]$Corpo) {
    $null = [Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime]
    $null = [Windows.Data.Xml.Dom.XmlDocument, Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime]
    $xml = New-Object Windows.Data.Xml.Dom.XmlDocument
    $xml.LoadXml('<toast><visual><binding template="ToastGeneric"><text/><text/></binding></visual></toast>')
    $textos = $xml.GetElementsByTagName('text')
    $null = $textos.Item(0).AppendChild($xml.CreateTextNode($Titulo))
    $null = $textos.Item(1).AppendChild($xml.CreateTextNode($Corpo))
    $aviso = [Windows.UI.Notifications.ToastNotification]::new($xml)
    [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($AppDoPowerShell).Show($aviso)
}

function Anotar([string]$Pasta, [string]$Texto) {
    # vigia.csv: prova de que o vigia rodou, com ou sem problema
    $arquivo = Join-Path $Pasta 'vigia.csv'
    if (-not (Test-Path -LiteralPath $arquivo)) { Set-Content -LiteralPath $arquivo -Value 'quando,problemas' -Encoding UTF8 }
    $linha = '{0},"{1}"' -f (Get-Date -Format 's'), $Texto.Replace('"', '""')
    Add-Content -LiteralPath $arquivo -Value $linha -Encoding UTF8
}

try {
    if (-not $SemEspera) { Esperar-Rotinas }
    $doLago = Ler-Rodadas (Join-Path $PastaLago 'runs.csv')
    $doPaper = Ler-Rodadas (Join-Path $PastaPaper 'runs.csv')
    $problemas = @(Problemas-DoDia $Hoje $doLago $doPaper (Ler-UltimoDoLago $PastaLago))
    $texto = $problemas -join '; '
    if ($SoMostrar) {
        if ($problemas.Count -eq 0) { 'vigia: tudo em dia' } else { $problemas | ForEach-Object { "vigia: $_" } }
    } else {
        Anotar $PastaPaper $texto
        if ($problemas.Count -gt 0) { Avisar "vigia do paper: $($problemas.Count) problema(s)" $texto }
    }
    if ($problemas.Count -gt 0) { exit 1 }
    exit 0
} catch {
    $erro = "o vigia falhou: $($_.Exception.Message)"
    if ($SoMostrar) { "vigia: $erro" } else { Avisar 'vigia do paper: falhou' $erro }
    exit 2
}
