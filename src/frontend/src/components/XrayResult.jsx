import React from "react";

const BoneAgeResult = ({ formData }) => {
  if (!formData) return <p>Carregando...</p>;

  const { idadePredita, preview, fileName, sexo } = formData;

  // Usa a idade em meses real (arredonda só para inteiro)
  const totalMeses = Math.round(idadePredita);
  const years = Math.floor(totalMeses / 12);
  const months = totalMeses % 12;

  // Texto amigável
  let idadeFormatada = `Aproximadamente ${years} anos`;
  if (months > 0) {
    idadeFormatada += ` e ${months} meses`;
  }

  return (
    <div className="uploadPanel">
      <h2>Resultado da Predição</h2>

      {preview && (
        <img
          src={preview}
          alt="Uploaded"
          className="preview-image"
          style={{ maxWidth: "300px", marginBottom: "20px" }}
        />
      )}

      <section className="result-section">
        {fileName && <p><strong>Arquivo:</strong> {fileName}</p>}
        <p><strong>Sexo:</strong> {sexo}</p>
        <p><strong>Idade prevista:</strong> {idadeFormatada}</p>
      </section>

      <button
        onClick={() => (window.location.href = "/")}
        className="submit active"
        style={{ marginTop: "20px" }}
      >
        Novo Upload
      </button>
    </div>
  );
};

export default BoneAgeResult;
