<#
.SYNOPSIS
Vigia das 12h do paper trading: confere se a rotina do lago e a do paper rodaram hoje e deram certo, e avisa no
Windows quando não.

.DESCRIPTION
Lê o runs.csv do lago (%LOCALAPPDATA%\b3lake, formato do b3lake) e o do paper (%LOCALAPPDATA%\paper-trading,
formato do anotar da rotina). Problemas: rotina sem linha hoje; última rodada de hoje do lago "failed" ou
"missing_overdue"; última de hoje do paper "falhou", "bloqueada" ou "lago rodando"; paper decidido num fechamento
mais velho que o que o lago já tinha publicado quando o paper rodou. Tarefa "paper vigia", seg-sex 12h. Antes de
conferir, espera as rotinas que estiverem rodando: com o PC ligado depois do horário, as tarefas atrasadas rodam
juntas.
Since 2026-10-23 (DECISOES 121) it also reads the paper-v2 runs.csv and the upkeep's (%LOCALAPPDATA%\paper-manutencao,
scripts/manutencao.py: no line today, or "falhou" / "bloqueada" / "lago rodando"), and warns when the app checkout
the tasks run is not on main. The notification itself is scripts/avisar.ps1.
Since 2026-10-23 (DECISOES 123) it also reads the lake's news/social log runs (%LOCALAPPDATA%\b3lake\
sentiment_runs.csv, task "b3lake sentimento", every 2 h): no "ok" or "partial" run on the day, or a day whose last
run "failed".
Since 2026-10-28 (DECISOES 166) it also reads the paper-v3 runs.csv (%LOCALAPPDATA%\paper-v3, scripts/paper_v3.py,
task "paper v3" at 10:30): no line today, or "falhou" / "bloqueada" / "lago rodando".

.EXAMPLE
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\vigia.ps1 -SoMostrar -SemEspera
#>
param(
    [string]$Hoje = '',     # o dia conferido; vazio = o da rodada perdida, pela hora de -Agora (Dia-Vigiado)
    [string]$Agora = (Get-Date -Format 's'),
    [string]$PastaLago = (Join-Path $env:LOCALAPPDATA 'b3lake'),
    [string]$PastaPaper = (Join-Path $env:LOCALAPPDATA 'paper-trading'),
    [string]$PastaPaperV2 = (Join-Path $env:LOCALAPPDATA 'paper-v2'),
    [string]$PastaPaperV3 = (Join-Path $env:LOCALAPPDATA 'paper-v3'),
    [string]$PastaManutencao = (Join-Path $env:LOCALAPPDATA 'paper-manutencao'),
    [string]$Repositorio = (Split-Path -Parent $PSScriptRoot),     # o checkout do app que as rotinas rodam
    [switch]$ManutencaoRodando,     # a manutenção ainda roda (teste; sem ele, pergunta à tarefa, fora do -SemEspera)
    [switch]$SoMostrar,     # escreve os problemas na saída em vez de avisar no Windows (teste e conferência à mão)
    [switch]$SemEspera      # não espera as rotinas que estiverem rodando (teste)
)

$ErrorActionPreference = 'Stop'

$RotinasVigiadas = @('b3lake daily', 'paper trading', 'paper v2', 'paper v3', 'paper manutencao')
$RamoDasRotinas = 'main'
$EsperaMaxima = [TimeSpan]::FromMinutes(45)
$FalhasDoLago = @('failed', 'missing_overdue')
$ProntosDoLago = @('done', 'up_to_date')
$FalhasDoPaper = @('falhou', 'bloqueada', 'lago rodando')

function Esperar-Rotinas {
    # rodada atrasada (PC ligado ou acordado depois do horário): as outras tarefas atrasadas começam junto com esta;
    # na rodada das 12h os 3 minutos não custam nada
    Start-Sleep -Seconds 180
    $limite = (Get-Date) + $EsperaMaxima
    while ((Get-Date) -lt $limite) {
        $rodando = @(Get-ScheduledTask -TaskName $RotinasVigiadas -ErrorAction SilentlyContinue |
                     Where-Object State -eq 'Running')
        if ($rodando.Count -eq 0) { return }
        Start-Sleep -Seconds 30
    }
}

function Dia-Vigiado([string]$Agora) {
    # rodada que caiu noutro dia (antes das 12h, ou no fim de semana): a perdida é a do dia útil anterior, e as
    # rotinas de hoje ainda podem nem ter rodado
    $momento = [datetime]::Parse($Agora, [Globalization.CultureInfo]::InvariantCulture)
    $fimDeSemana = @('Saturday', 'Sunday')
    $dia = $momento.Date
    if ($momento.TimeOfDay -lt [TimeSpan]'12:00:00' -or $fimDeSemana -contains "$($dia.DayOfWeek)") {
        do { $dia = $dia.AddDays(-1) } while ($fimDeSemana -contains "$($dia.DayOfWeek)")
    }
    return $dia.ToString('yyyy-MM-dd')
}

function Ler-Rodadas([string]$Arquivo) {
    if (-not (Test-Path -LiteralPath $Arquivo)) { return @() }
    return @(Import-Csv -LiteralPath $Arquivo -Encoding UTF8)
}

function Problemas-DoDia([string]$Dia, [object[]]$DoLago, [object[]]$DoPaper, [object[]]$DoPaperV2,
                         [object[]]$DoManutencao, [bool]$ManutencaoAindaRoda, [object[]]$DoSentimento,
                         [object[]]$DoPaperV3) {
    $problemas = New-Object System.Collections.Generic.List[string]
    # log de notícias (b3lake.sentiment, a cada 2 h, todo dia): ao menos uma rodada certa no dia (UTC do started)
    $sentimentoHoje = @($DoSentimento | Where-Object { "$($_.started)".StartsWith($Dia) })
    $certas = @($sentimentoHoje | Where-Object { @('ok', 'partial') -contains $_.outcome })
    if ($sentimentoHoje.Count -gt 0 -and $sentimentoHoje[-1].outcome -eq 'failed') {
        $problemas.Add("log de notícias: failed ($($sentimentoHoje[-1].detail))")
    } elseif ($certas.Count -eq 0) {
        $problemas.Add('log de notícias: nenhuma rodada certa hoje (tarefa "b3lake sentimento")')
    }
    # manutenção (scripts/manutencao.py): rodou e não falhou nem ficou bloqueada; ainda rodando (leitura longa depois
    # de esperar as outras rotinas) não é falta
    $manutencaoHoje = @($DoManutencao | Where-Object { "$($_.quando)".StartsWith($Dia) })
    if ($manutencaoHoje.Count -eq 0 -and -not $ManutencaoAindaRoda) {
        $problemas.Add('manutenção: nenhuma rodada hoje (tarefa "paper manutencao")')
    } elseif ($FalhasDoPaper -contains $manutencaoHoje[-1].desfecho) {
        $problemas.Add("manutenção: $($manutencaoHoje[-1].desfecho) ($($manutencaoHoje[-1].nota))")
    }
    # paper-v2: só rodou e não falhou (o fechamento vem do mesmo precos_diarios do paper, conferido abaixo)
    $v2Hoje = @($DoPaperV2 | Where-Object { "$($_.quando)".StartsWith($Dia) })
    if ($v2Hoje.Count -eq 0) {
        $problemas.Add('paper v2: nenhuma rodada hoje (tarefa "paper v2")')
    } elseif ($FalhasDoPaper -contains $v2Hoje[-1].desfecho) {
        $problemas.Add("paper v2: $($v2Hoje[-1].desfecho) ($($v2Hoje[-1].nota))")
    }
    # paper-v3: the same check as paper-v2
    $v3Hoje = @($DoPaperV3 | Where-Object { "$($_.quando)".StartsWith($Dia) })
    if ($v3Hoje.Count -eq 0) {
        $problemas.Add('paper v3: nenhuma rodada hoje (tarefa "paper v3")')
    } elseif ($FalhasDoPaper -contains $v3Hoje[-1].desfecho) {
        $problemas.Add("paper v3: $($v3Hoje[-1].desfecho) ($($v3Hoje[-1].nota))")
    }
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
    if ($paperHoje.Count -eq 0) { return $problemas }
    # o fechamento que o paper podia ver: o maior "expected" das rodadas certas do lago que começaram antes da rodada
    # de hoje do paper (o state.json não serve: depois das 21h45 ele já tem o fechamento de hoje)
    $inicioDoPaper = "$($paperHoje[-1].inicio)"
    $publicados = @($DoLago | Where-Object { $ProntosDoLago -contains $_.outcome -and "$($_.started)" -lt $inicioDoPaper } |
                    ForEach-Object { "$($_.expected)" } | Sort-Object)
    if ($publicados.Count -eq 0) { return $problemas }
    $doLago = $publicados[-1]
    $fechamentos = @($DoPaper | ForEach-Object { "$($_.fechamento)" } | Where-Object { $_ } | Sort-Object)
    $ultimoDoPaper = if ($fechamentos.Count -gt 0) { $fechamentos[-1] } else { $null }
    if (-not $ultimoDoPaper -or $ultimoDoPaper -lt $doLago) {
        $problemas.Add("paper trading: último fechamento decidido $ultimoDoPaper, o lago já tem o de $doLago")
    }
    return $problemas
}

function Problema-DoRamo([string]$Repositorio) {
    # as tarefas rodam este checkout: noutro ramo, o paper-v1 não registrou o dia (22/10/2026)
    $ramo = & git -C $Repositorio symbolic-ref -q --short HEAD
    if ($ramo -eq $RamoDasRotinas) { return $null }
    $onde = if ($ramo) { "em $ramo" } else { 'com HEAD destacado' }
    return "checkout do app $onde, as rotinas rodam na $RamoDasRotinas (git switch $RamoDasRotinas)"
}

function Avisar([string]$Titulo, [string]$Corpo) {
    # o mesmo aviso da manutenção (scripts/avisar.ps1)
    & (Join-Path $PSScriptRoot 'avisar.ps1') -Titulo $Titulo -Corpo $Corpo
}

function Anotar([string]$Pasta, [string]$Texto) {
    # vigia.csv: prova de que o vigia rodou, com ou sem problema
    $null = New-Item -ItemType Directory -Force -Path $Pasta      # máquina nova: a pasta do paper pode não existir
    $arquivo = Join-Path $Pasta 'vigia.csv'
    if (-not (Test-Path -LiteralPath $arquivo)) { Set-Content -LiteralPath $arquivo -Value 'quando,problemas' -Encoding UTF8 }
    $linha = '{0},"{1}"' -f (Get-Date -Format 's'), $Texto.Replace('"', '""')
    Add-Content -LiteralPath $arquivo -Value $linha -Encoding UTF8
}

try {
    # sem console (tarefa agendada escondida) o setter pode falhar: a saída só importa no -SoMostrar
    try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }
    if (-not $SemEspera) { Esperar-Rotinas }
    $doLago = Ler-Rodadas (Join-Path $PastaLago 'runs.csv')
    $doPaper = Ler-Rodadas (Join-Path $PastaPaper 'runs.csv')
    $doPaperV2 = Ler-Rodadas (Join-Path $PastaPaperV2 'runs.csv')
    $doPaperV3 = Ler-Rodadas (Join-Path $PastaPaperV3 'runs.csv')
    $doManutencao = Ler-Rodadas (Join-Path $PastaManutencao 'runs.csv')
    $doSentimento = Ler-Rodadas (Join-Path $PastaLago 'sentiment_runs.csv')
    $dia = if ($Hoje) { $Hoje } else { Dia-Vigiado $Agora }
    $daManutencao = @(Get-ScheduledTask -TaskName 'paper manutencao' -ErrorAction SilentlyContinue)
    $tarefaRodando = -not $SemEspera -and @($daManutencao | Where-Object State -eq 'Running').Count -gt 0
    $aindaRoda = $ManutencaoRodando.IsPresent -or $tarefaRodando
    $problemas = @(Problemas-DoDia $dia $doLago $doPaper $doPaperV2 $doManutencao $aindaRoda $doSentimento $doPaperV3)
    $doRamo = Problema-DoRamo $Repositorio
    if ($doRamo) { $problemas += $doRamo }
    $texto = $problemas -join '; '
    if ($SoMostrar) {
        if ($problemas.Count -eq 0) { 'vigia: tudo em dia' } else { $problemas | ForEach-Object { "vigia: $_" } }
    } else {
        if ($problemas.Count -gt 0) { Avisar "vigia do paper: $($problemas.Count) problema(s)" $texto }
        Anotar $PastaPaper "$dia $texto"
    }
    if ($problemas.Count -gt 0) { exit 1 }
    exit 0
} catch {
    $erro = "o vigia falhou: $($_.Exception.Message)"
    if ($SoMostrar) { "vigia: $erro" } else { Avisar 'vigia do paper: falhou' $erro }
    exit 2
}
