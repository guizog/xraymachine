import './App.css';

import FileUpload from './components/FileUpload.jsx';
import Form from "./Form.jsx";

function App() {
    return (
        <>
            <div className="uploadPanel">
                <h1>Upload your x-ray image</h1>
                <Form/>
            </div>
        </>
    );
}

export default App;