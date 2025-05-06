import React from "react";

const XrayResult = ({formData}) => {



    return (
        <>
            <section>
                X-ray results:
                <ul>
                    <li>Xray: ###Show image in here#####</li>
                    <li>Expect bone age: {formData.results.boneAge}</li>
                </ul>
            </section>
        </>
    )
}

export default XrayResult