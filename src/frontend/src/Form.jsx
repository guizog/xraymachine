import FileUpload from './components/FileUpload.jsx';
import {useState} from "react";
import XrayResult from "./components/XrayResult.jsx";

function Form() {
    const [isSubmitted, setIsSubmitted] = useState(null);
    const [formData, setFormData] = useState(null);

    return (
        <>
            {!isSubmitted && <FileUpload setIsSubmitted={setIsSubmitted} setFormData={setFormData}/>}
            {isSubmitted && <XrayResult formData={formData}/>}
        </>
    );
}

export default Form;