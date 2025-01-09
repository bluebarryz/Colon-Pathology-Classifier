import { useState } from "react";
import styles from "./PathologyClassifier.module.css";

const PathologyClassifier = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleFileSelect = (event) => {
        const file = event.target.files[0];
        if (file) setSelectedFile(file);
    };

    const handleSubmit = async () => {
        if (!selectedFile) return;

        try {
            setLoading(true);
            const formData = new FormData();
            formData.append("image", selectedFile);

            const response = await fetch("http://localhost:5000/predict", {
                method: "POST",
                body: formData,
            });

            const data = await response.json();
            setPrediction(data.prediction);
        } catch (err) {
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className={styles.container}>
            <h1>PathMNIST Image Classifier</h1>
            <p>Upload an image to classify it</p>

            <input type="file" accept="image/*" onChange={handleFileSelect} className={styles.button} />

            <button onClick={handleSubmit} disabled={loading || !selectedFile} className={styles.button}>
                {loading ? "Processing..." : "Classify Image"}
            </button>

            {prediction !== null && <p>Prediction: {prediction}</p>}
        </div>
    );
};

export default PathologyClassifier;
