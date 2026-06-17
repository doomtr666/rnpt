# Summarize rnpt-bench criterion results as a Mray/s table.
# Usage: .\scripts\bench_summary.ps1

$base = "$PSScriptRoot\..\target\criterion"
if (-not (Test-Path $base)) {
    Write-Error "No criterion results found. Run: cargo bench -p rnpt-bench --features embree"
    exit 1
}

$results = @{}
Get-ChildItem "$base" -Recurse -Filter "estimates.json" |
    Where-Object { $_.FullName -like "*\new\*" } |
    ForEach-Object {
        $rel   = $_.FullName.Substring($base.Length + 1)
        $parts = $rel.Split("\")
        if ($parts.Count -ge 4) {
            $group = $parts[0]; $impl = $parts[1]; $size = $parts[2]
            $d     = Get-Content $_.FullName -Raw | ConvertFrom-Json
            $mrays = [math]::Round(1024 / ($d.mean.point_estimate / 1e9) / 1e6, 1)
            $results["$group|$impl|$size"] = $mrays
        }
    }

$scenes = @("hf", "cluster", "soup")
$rays   = @("coherent", "incoherent", "shadow")
$sizes  = @("10k", "100k", "1m")
$impls  = @("rnpt", "embree", "embree8")

# Collect all impl names that actually have data
$activeImpls = $impls | Where-Object { $i = $_; $results.Keys | Where-Object { $_ -like "*|$i|*" } }

$hdr = "{0,-26}" -f "Mray/s"
foreach ($s in $sizes) { foreach ($i in $activeImpls) { $hdr += " {0,8}" -f "$i/$s" } }
Write-Host $hdr
Write-Host ("-" * $hdr.Length)

foreach ($ray in $rays) {
    foreach ($scene in $scenes) {
        $g   = "${ray}_${scene}"
        $row = "{0,-26}" -f "$ray/$scene"
        foreach ($s in $sizes) {
            foreach ($i in $activeImpls) {
                $v = if ($results.ContainsKey("$g|$i|$s")) { $results["$g|$i|$s"] } else { "-" }
                $row += " {0,8}" -f $v
            }
        }
        Write-Host $row
    }
    Write-Host ("-" * $hdr.Length)
}
