# Re-runs every notebook in order, then the full test suite, then checks the
# key results. Run from the repo root with the project's venv active:
#
#   .\.venv\Scripts\Activate.ps1
#   .\scripts\run_all.ps1              # everything
#   .\scripts\run_all.ps1 -StartAt 9   # resume from notebook 09 after a failure
#   .\scripts\run_all.ps1 -SkipNotebooks   # tests and checks only
#
# Notebook outputs are saved in place, so the committed notebooks show the
# results from your machine.

param(
    [int]$StartAt = 1,
    [switch]$SkipNotebooks
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONPATH = "src"

# --- environment checks ------------------------------------------------------
if (-not $env:VIRTUAL_ENV) {
    Write-Host "No virtual environment is active. Run .\.venv\Scripts\Activate.ps1 first." -ForegroundColor Red
    exit 1
}
$pyver = python -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"
if ($pyver -ne "3.11") {
    Write-Host "Python $pyver is active; the project (and the pinned live-demo models) use 3.11." -ForegroundColor Red
    exit 1
}
foreach ($f in "models\fd001\rsf.joblib", "models\fd004\rsf.joblib", "models\fd001\drift_reference.json", "models\fd004\drift_reference.json") {
    if (-not (Test-Path $f)) {
        Write-Host "$f is missing. Export it first:" -ForegroundColor Red
        Write-Host "  python -m turbofan_rul.export_deployment_artifacts --subset FD001"
        Write-Host "  python -m turbofan_rul.export_deployment_artifacts --subset FD004"
        Write-Host "(or, for models exported earlier, just: python -m turbofan_rul.drift --subset FD001 --subset FD004)"
        exit 1
    }
}

$total = [Diagnostics.Stopwatch]::StartNew()

# --- notebooks ---------------------------------------------------------------
if (-not $SkipNotebooks) {
    # Notebooks 15, 16, 18 and 19 load their out-of-fold predictions from these files
    # when they exist. Deleting them makes the notebooks recompute them from
    # scratch (~10-40 minutes each), so the selections are reproduced on this
    # machine, not just re-read.
    if ($StartAt -le 15) {
        Remove-Item data\processed\interval_oof_fd001.csv, data\processed\interval_oof_fd004.csv -ErrorAction SilentlyContinue
    }
    if ($StartAt -le 16) {
        Remove-Item data\processed\interval_oof_uncapped_*.csv -ErrorAction SilentlyContinue
    }
    if ($StartAt -le 18) {
        Remove-Item data\processed\point_candidates_oof_fd001.csv -ErrorAction SilentlyContinue
    }
    if ($StartAt -le 19) {
        Remove-Item data\processed\point_candidates_oof_fd004.csv -ErrorAction SilentlyContinue
    }

    # -Filter doesn't support [0-9] ranges, so match the "NN_" prefix with a regex.
    $notebooks = Get-ChildItem notebooks -Filter "*.ipynb" |
        Where-Object { $_.Name -match '^\d{2}_[A-Za-z0-9_]+\.ipynb$' -and [int]$_.Name.Substring(0, 2) -ge $StartAt } |
        Sort-Object Name
    if (-not $notebooks) {
        Write-Host "No notebooks found to run (looked for notebooks\NN_*.ipynb from $StartAt)." -ForegroundColor Red
        exit 1
    }
    Write-Host "Running $($notebooks.Count) notebooks: $($notebooks[0].Name) ... $($notebooks[-1].Name)"

    foreach ($nb in $notebooks) {
        $n = [int]$nb.Name.Substring(0, 2)
        Write-Host "`n=== $($nb.Name)" -ForegroundColor Cyan
        $sw = [Diagnostics.Stopwatch]::StartNew()
        python -m jupyter nbconvert --to notebook --execute --inplace `
            --ExecutePreprocessor.timeout=-1 $nb.FullName
        if ($LASTEXITCODE -ne 0) {
            Write-Host "`nFAILED: $($nb.Name). Fix it, then resume with:" -ForegroundColor Red
            Write-Host "  .\scripts\run_all.ps1 -StartAt $n"
            exit 1
        }
        Write-Host ("    done in {0:N1} min" -f $sw.Elapsed.TotalMinutes) -ForegroundColor Green
    }
}

# --- tests (including the embedded-mode dashboard tests) ---------------------
Write-Host "`n=== pytest" -ForegroundColor Cyan
$env:MODELS_DIR = "models"
python -m pytest tests -q
$testsExit = $LASTEXITCODE
Remove-Item Env:MODELS_DIR
if ($testsExit -ne 0) {
    Write-Host "`nTests FAILED -- paste the output." -ForegroundColor Red
    exit 1
}

# --- key results -------------------------------------------------------------
Write-Host "`n=== key results" -ForegroundColor Cyan
# Single quotes only inside the Python code: Windows PowerShell strips double
# quotes from arguments passed to native programs.
python -c @'
import json
from turbofan_rul import manifest
for s in ('fd001', 'fd004'):
    m = json.load(open('models/' + s + '/deployment_metadata.json'))
    version, stale = manifest.verified_version('models/' + s)
    print('  ' + s + ': RSF interval correction q = %.2f, model version %s' % (m['rsf_interval_correction'], version))
    if stale:
        print('      (no up-to-date manifest -- run: python -m turbofan_rul.manifest --subset ' + s.upper() + ')')
'@
if ($LASTEXITCODE -ne 0) {
    Write-Host "Could not read the deployment metadata." -ForegroundColor Red
    exit 1
}
$checks = @(
    @{ File = "notebooks\07_deep_survival_stretch.ipynb";   Text = "Deep runs matching or beating RSF on test C-index, NASA or RMSE: 0 of 6" },
    @{ File = "notebooks\14_interval_coverage_check.ipynb"; Text = "FD001 reproduces notebook 12" },
    @{ File = "notebooks\15_interval_fix.ipynb";            Text = "Selected by the rule: hybrid+none" },
    @{ File = "notebooks\15_interval_fix.ipynb";            Text = "Confirmation rule: PASSED" },
    @{ File = "notebooks\16_uncapped_upper_bound.ipynb";    Text = "better on C-index and RMSE on both subsets: False" },
    @{ File = "notebooks\16_uncapped_upper_bound.ipynb";    Text = "Selected within the 250 MB budget: uncapped RSF upper (leaf 30)" },
    @{ File = "notebooks\16_uncapped_upper_bound.ipynb";    Text = "Confirmation on the test sets: PASSED" },
    @{ File = "notebooks\17_drift_monitoring.ipynb";         Text = "Own fleets: no drift windows" },
    @{ File = "notebooks\18_point_estimate_candidates.ipynb"; Text = "Selected on cross-validation: Blend RSF + GB, cap 125" },
    @{ File = "notebooks\18_point_estimate_candidates.ipynb"; Text = "Test-set rule -- blend NASA lower than deployed: PASSED" },
    @{ File = "notebooks\18_point_estimate_candidates.ipynb"; Text = "Blend has no more large late errors (>20 and >30 cycles) than RSF alone: True" },
    @{ File = "notebooks\19_fd004_point_estimate.ipynb";     Text = "Selected on cross-validation: none -- the deployed RSF stays" },
    @{ File = "notebooks\19_fd004_point_estimate.ipynb";     Text = "Blend better at <= 150 cycles left in 5 of 5 folds" }
)
$ok = $true
foreach ($c in $checks) {
    if (Select-String -Path $c.File -SimpleMatch $c.Text -Quiet) {
        Write-Host "  ok   $($c.Text)" -ForegroundColor Green
    } else {
        Write-Host "  MISSING   '$($c.Text)' in $($c.File)" -ForegroundColor Red
        $ok = $false
    }
}

Write-Host ("`nTotal time: {0:N0} min" -f $total.Elapsed.TotalMinutes)
if ($ok) { Write-Host "All notebooks, tests and checks passed." -ForegroundColor Green } else { exit 1 }
