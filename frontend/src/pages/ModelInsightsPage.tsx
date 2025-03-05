import { useEffect } from "react";
// import { motion } from "framer-motion";
// import SectionHeading from "../components/SectionHeading";
// import AccordionItem from "../components/AccordionItem";
// import ModelCard from "../components/ModelCard";
// import ModelDiagram from "../components/ModelDiagram";

// Sample model information
const models = [
	{
		id: "cnn",
		name: "CNN",
		fullName: "Convolutional Neural Network",
		type: "Deep Learning",
		purpose: "Image Classification",
		description:
			"Our CNN model specializes in feature extraction from satellite imagery, identifying shoreline boundaries with high precision.",
		metrics: {
			accuracy: 92,
			precision: 94,
			recall: 90,
			f1: 92,
		},
		color: "#0092ca",
	},
	{
		id: "lstm",
		name: "LSTM",
		fullName: "Long Short-Term Memory",
		type: "Deep Learning",
		purpose: "Time Series Prediction",
		description:
			"The LSTM model analyzes historical shoreline data to predict future changes and erosion patterns over time.",
		metrics: {
			accuracy: 88,
			precision: 86,
			recall: 85,
			f1: 85.5,
		},
		color: "#7E69AB",
	},
	{
		id: "rf",
		name: "Random Forest",
		fullName: "Random Forest",
		type: "Ensemble Learning",
		purpose: "Feature Classification",
		description:
			"Our Random Forest classifier helps identify key environmental factors contributing to shoreline changes.",
		metrics: {
			accuracy: 85,
			precision: 83,
			recall: 82,
			f1: 82.5,
		},
		color: "#F97316",
	},
];

const ModelInsightsPage = () => {
	useEffect(() => {
		window.scrollTo(0, 0);
	}, []);

	return (
		<div className="pt-32 pb-24">
			
		</div>
	);
};

export default ModelInsightsPage;
