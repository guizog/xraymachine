import React, {useState} from 'react';

const FileUpload = ({setIsSubmitted, setFormData}) => {
    const [file, setFile] = useState(null);

    const handleFileChange = (e) => {
        if (e.target.files) {
            setFile(e.target.files[0]);
        }
    };

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
        <>
            <div className="input-group">
                <input id="file" type="file" onChange={handleFileChange}/>
            </div>
            {file && (
                <section>
                    File details:
                    <ul>
                        <li>Name: {file.name}</li>
                        <li>Type: {file.type}</li>
                        <li>Size: {file.size} bytes</li>
                    </ul>
                </section>
            )}

            {!file && (
                <button disabled
                    className="submit"
                >Select a file to upload</button>
            )}
            {file && (
                <button
                    onClick={handleUpload}
                    className="submit"
                >Upload file</button>
            )}
        </>
    );
};

export default FileUpload;