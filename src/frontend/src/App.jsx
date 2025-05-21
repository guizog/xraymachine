import './App.css';

import FileUpload from './components/FileUpload.jsx';
import Form from "./Form.jsx";

function App() {
    return (
        <>
            <div>
                <div className="logo">
                    <i className="fa-solid fa-user-doctor fa-8x"></i>
                    <h1>TCC Analise de idade óssea via IA</h1>
                </div>
                <br></br>
                <Form/>
            </div>
        </>
    );
}

export default App;