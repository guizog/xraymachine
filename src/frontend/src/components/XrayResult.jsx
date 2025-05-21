import React, {useState} from "react";

const XrayResult = ({formData}) => {
    return (
        <div className="uploadPanel">
            <section className="file-section">
                <p className="section-title">X-ray results:</p>
                <ul className="file-info">
                    <li><strong>Uploaded Xray image:</strong> #### WIP ####</li>
                    <li><strong>Expect bone age:</strong> {formData.results.boneAge}</li>
                </ul>
            </section>
        </div>
    )
}

export default XrayResult