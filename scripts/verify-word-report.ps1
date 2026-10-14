param([Parameter(Mandatory=$true)][string]$DocumentPath, [Parameter(Mandatory=$true)][string]$PdfPath)
$ErrorActionPreference = 'Stop'
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
$document = $null
try {
    $document = $word.Documents.Open((Resolve-Path -LiteralPath $DocumentPath).Path, $false, $false)
    $document.Repaginate()
    foreach ($toc in $document.TablesOfContents) { $toc.Update() }
    for ($pass = 0; $pass -lt 3; $pass++) {
        $document.Repaginate()
        foreach ($toc in $document.TablesOfContents) { $toc.UpdatePageNumbers() }
    }
    $document.Save()
    $document.ExportAsFixedFormat([IO.Path]::GetFullPath($PdfPath), 17)
    Write-Output ('pages=' + $document.ComputeStatistics(2))
    Write-Output ('toc_count=' + $document.TablesOfContents.Count)
} finally {
    if ($null -ne $document) { $document.Close($false) }
    $word.Quit()
    [Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
