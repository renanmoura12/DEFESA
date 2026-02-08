public bool ExportExcell()
{
    var examesSelecionados = GetValoresByExameNames(connection);

    var examesObstetricos = SepararExames(examesSelecionados, "OBSTETRICA");

    var examesPorConsulta = GetExamesPorConsulta(connection, examesSelecionados);
    var examesPorConsultaObst = GetExamesPorConsulta(connection, examesObstetricos);

    MesclarConsultas(examesPorConsulta, examesPorConsultaObst);

    var laudos = GetLaudos(connection, examesPorConsulta);

    var obstetricas = ParseLaudos<ObstetricaDto>(laudos, "OBSTETRICA", CorrigirJsonObstetrica);
    var primeiroTrimestre = ParseLaudos<PrimeiroTrimestreDto>(laudos, "OBSTETRICA 1a TRIMESTRE");
    var translucenciaNucal = ParseLaudos<TranslucenciaNucalDto>(laudos, "TRANSLUCENCIA NUCAL", CorrigirJsonTN);

    var unifiedExports = UnificarExports(obstetricas, primeiroTrimestre, translucenciaNucal);

    EnriquecerComQuestionarios(unifiedExports);

    SalvarExcel(unifiedExports);

    return true;
}

private List<ExameDto> SepararExames(List<ExameDto> exames, string nome)
{
    var selecionados = exames.Where(e => e.ExameNome == nome).ToList();
    exames.RemoveAll(e => e.ExameNome == nome);
    return selecionados;
}

private void MesclarConsultas(
    List<ExameConsultaDto> principal,
    List<ExameConsultaDto> obstetricas)
{
    foreach (var item in obstetricas)
    {
        if (!principal.Any(p => p.PacienteId == item.PacienteId))
            principal.Add(item);
    }
}

private List<T> ParseLaudos<T>(
    IEnumerable<LaudoDto> laudos,
    string nomeExame,
    Func<string, string>? normalizador = null)
    where T : class, new()
{
    var resultado = new List<T>();

    var filtrados = laudos.Where(l => l.NomeExamePrincipal == nomeExame);

    foreach (var item in filtrados)
    {
        foreach (var json in ExtrairJsons(item.Laudo, normalizador))
        {
            try
            {
                var dto = JsonSerializer.Deserialize<T>(json);
                if (dto == null) continue;

                PreencherCamposPadrao(dto, item);
                resultado.Add(dto);
            }
            catch { }
        }
    }

    return resultado;
}

private IEnumerable<string> ExtrairJsons(string laudo, Func<string, string>? normalizador)
{
    var texto = laudo.Trim('[', ']');

    if (normalizador != null)
        texto = normalizador(texto);

    return Regex.Split(texto, @"(?<=\})\s*,\s*(?=\{)");
}

private string CorrigirJsonTN(string json) =>
    json.Replace("\"perdasGestacionais\": ,", "");

private string CorrigirJsonObstetrica(string json) =>
    json.Replace("\"gestacaoConclusao3\": ,", "");

private void PreencherCamposPadrao<T>(T dto, LaudoDto item)
{
    typeof(T).GetProperty("Data")?.SetValue(dto, item.Data.ToString("dd/MM/yyyy"));
    typeof(T).GetProperty("PacienteId")?.SetValue(dto, item.PacienteId);
}

private List<ExportDto> UnificarExports(
    List<ObstetricaDto> obst,
    List<PrimeiroTrimestreDto> primeiro,
    List<TranslucenciaNucalDto> tn)
{
    var lista = new List<ExportDto>();

    lista.AddRange(Mapear<ObstetricaDto>(obst));
    lista.AddRange(Mapear<PrimeiroTrimestreDto>(primeiro));
    lista.AddRange(Mapear<TranslucenciaNucalDto>(tn));

    return lista;
}

private IEnumerable<ExportDto> Mapear<T>(IEnumerable<T> origem)
{
    return origem.Select(o =>
        JsonSerializer.Deserialize<ExportDto>(
            JsonSerializer.Serialize(o)));
}

private void EnriquecerComQuestionarios(List<ExportDto> exports)
{
    var questionarios = GetQuestionarios(connection);
    var dict = questionarios
        .GroupBy(q => q.PacienteId)
        .ToDictionary(g => g.Key, g => g.First());

    exports.RemoveAll(e => !dict.ContainsKey(e.PacienteId));

    exports
        .Sort((a, b) =>
            a.PacienteId != b.PacienteId
                ? a.PacienteId.CompareTo(b.PacienteId)
                : DateTime.ParseExact(b.Data, "dd/MM/yyyy", CultureInfo.InvariantCulture)
                    .CompareTo(DateTime.ParseExact(a.Data, "dd/MM/yyyy", CultureInfo.InvariantCulture)));

    int? pacienteAnterior = null;

    foreach (var export in exports)
    {
        if (export.PacienteId != pacienteAnterior &&
            dict.TryGetValue(export.PacienteId, out var q))
        {
            MapearQuestionario(export, q);
            pacienteAnterior = export.PacienteId;
        }
    }
}

private void SalvarExcel(List<ExportDto> dados)
{
    var folder = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Exports");
    Directory.CreateDirectory(folder);

    var path = Path.Combine(folder, "ExportFinal.xlsx");

    using var package = new ExcelPackage();
    var sheet = package.Workbook.Worksheets.Add("Unificado");

    sheet.Cells["A1"].LoadFromCollection(dados, true);
    package.SaveAs(new FileInfo(path));
}
