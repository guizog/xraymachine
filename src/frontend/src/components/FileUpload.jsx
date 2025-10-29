import React, { useEffect, useState } from "react";

const FileUpload = ({ setIsSubmitted, setFormData }) => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [sexo, setSexo] = useState(""); // agora começa vazio
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Atualiza preview da imagem
  useEffect(() => {
    if (!file) {
      setPreview(null);
      return;
    }
    const objectUrl = URL.createObjectURL(file);
    setPreview(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files?.[0];
    setFile(selectedFile || null);
    setError(""); // limpa erros ao trocar arquivo
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Selecione um arquivo antes de enviar.");
      return;
    }
    if (!sexo) {
      setError("Selecione o sexo antes de enviar.");
      return;
    }

    setLoading(true);
    setError("");

    const formData = new FormData();
    formData.append("file", file);
    formData.append("sexo", sexo);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Erro na API: ${response.status}`);
      }

      const data = await response.json();

      setFormData({
        idadePredita: data.idade_predita_meses,
        preview,
        fileName: file.name,
        sexo: data.sexo,
      });

      setIsSubmitted(true);
    } catch (err) {
      setError("Falha ao enviar arquivo. Tente novamente.");
      console.error(err);
      setIsSubmitted(false);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="uploadPanel">
      <h1 className="title">Predição de Idade Óssea</h1>

      <div className="input-group">
        <label>Selecione a imagem:</label>
        <input type="file" onChange={handleFileChange} accept="image/*" />
      </div>

      <div className="input-group">
        <label>Sexo:</label>
        <select value={sexo} onChange={(e) => setSexo(e.target.value)}>
          <option value="">-- Selecione --</option>
          <option value="F">Feminino</option>
          <option value="M">Masculino</option>
        </select>
      </div>

      {file && (
        <>
          <section className="file-section">
            <h2 className="section-title">Detalhes do Arquivo</h2>
            <ul className="file-info">
              <li><strong>Nome:</strong> {file.name}</li>
              <li><strong>Tipo:</strong> {file.type}</li>
              <li><strong>Tamanho:</strong> {file.size.toLocaleString()} bytes</li>
            </ul>
          </section>

          <h2 className="section-title">Pré-visualização da Imagem</h2>
          <img className="preview-image" src={preview} alt="Preview" />
        </>
      )}

      {error && <p className="error-message">{error}</p>}

      <button
        onClick={handleUpload}
        className={`submit ${file && sexo && !loading ? "active" : "inactive"}`}
        disabled={!file || !sexo || loading}
      >
        {loading ? "Enviando..." : "Enviar para Predição"}
      </button>
    </div>
  );
};

export default FileUpload;
