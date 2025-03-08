import React from "react";
import model1 from "../assets/images/model1.png";
import model2 from "../assets/images/model2.png";
import model3 from "../assets/images/model3.png";
import model4 from "../assets/images/model4.png";

// Define TypeScript interface for Model
interface Model {
	id: string;
	name: string;
	fullNameOverview: string;
	descriptionOverview: string;
	type: string;
	purpose: string;
	color: string;
}

// Define props type
interface ModelTableProps {
	models: Model[];
}

// Mapping Model IDs to their respective images
const modelImages: Record<string, string> = {
	model1,
	model2,
	model3,
	model4,
};

const ModelTable: React.FC<ModelTableProps> = ({ models }) => {
	return (
		<section className="mt-24">
			<h2 className="text-3xl font-bold text-center mb-8">Model Overview</h2>
			<div className="grid grid-cols-1 md:grid-cols-2 gap-8">
				{models.map((model) => (
					<div
						key={model.id}
						className="flex flex-col items-center bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md"
					>
						{/* Model Image */}
						<img
							src={modelImages[model.id] || modelImages["model1"]} // Fallback image handling
							alt={`${model.name} Visualization`}
							className="w-full h-full object-cover rounded-md"
						/>

						{/* Model Details */}
						<div className="mt-4 text-center">
							<h3 className="text-xl font-semibold text-shoreline-primary">
								{model.fullNameOverview}
							</h3>
							<p className="text-gray-600 dark:text-gray-300 mt-2">
								{model.descriptionOverview}
							</p>
						</div>
					</div>
				))}
			</div>
		</section>
	);
};

export default ModelTable;
