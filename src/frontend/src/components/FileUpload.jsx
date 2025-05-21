import React, {useEffect, useState} from 'react';

const FileUpload = ({setIsSubmitted, setFormData}) => {
    const [file, setFile] = useState(null);
    const [preview, setPreview] = useState(null);

    const handleFileChange = (e) => {
        if (e.target.files) {
            setFile(e.target.files[0]);
        }
    };

    useEffect(() => {
        if (!file) {
            setPreview(undefined);
            return;
        }

        const objectUrl = URL.createObjectURL(file);
        setPreview(objectUrl);

        return () => URL.revokeObjectURL(file);
    }, [file]);

    const handleUpload = async () => {
        if (file) {
            console.log("uploading file...");
            const formData = new FormData();
            formData.append("file", file);

            try {
                const result = await fetch('http://localhost:8000/api/uploadFile', {
                    method: "POST",
                    body: formData
                })
                const data = await result.json()

                setIsSubmitted(true);
                setFormData(data);
                console.log(data);
            } catch (error) {
                setIsSubmitted(false);
                console.log(error);
            }
        }
    };

    return (
        <div className="uploadPanel">
            <h1 className="title">Upload your x-ray image</h1>

            <div className="input-group">
                <input id="file" type="file" onChange={handleFileChange}/>
            </div>

            {file ? (
                <>
                    <section className="file-section">
                        <h2 className="section-title">File Details</h2>
                        <ul className="file-info">
                            <li><strong>Name:</strong> {file.name}</li>
                            <li><strong>Type:</strong> {file.type}</li>
                            <li><strong>Size:</strong> {file.size.toLocaleString()} bytes</li>
                        </ul>
                    </section>

                    <h2 className="section-title">Image Preview</h2>
                    <img className="preview-image" src={preview} alt="Preview"/>

                    <button onClick={handleUpload} className="submit active">Upload File</button>
                </>
            ) : (
                <button className="submit inactive" disabled>Select a file to upload</button>
            )}
        </div>
    );
};

export default FileUpload;