import { motion } from "framer-motion";
import { useState } from "react";

interface ModelMetrics {
	accuracy: number;
	precision: number;
	recall: number;
	f1: number;
}

interface Model {
	id: string;
	name: string;
	fullName: string;
	type: string;
	purpose: string;
	description: string;
	metrics: ModelMetrics;
	color: string;
}

interface ModelCardProps {
	model: Model;
}

const CustomProgress = ({ value, color }: { value: number; color: string }) => (
	<div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
		<div
			className="h-full rounded-full transition-all duration-300"
			style={{
				width: `${value}%`,
				backgroundColor: color,
			}}
		/>
	</div>
);

const ModelCard = ({ model }: ModelCardProps) => {
	const [isHovered, setIsHovered] = useState(false);

	return (
		<motion.div
			initial={{ opacity: 0, y: 20 }}
			whileInView={{ opacity: 1, y: 0 }}
			transition={{ duration: 0.5 }}
			viewport={{ once: true, margin: "-100px" }}
			className={`
        glossy-card p-6 border-2 transition-all
        bg-white dark:bg-gray-800
        ${
			isHovered
				? `border-[${model.color}] shadow-lg dark:shadow-[${model.color}]/20`
				: "border-transparent hover:border-gray-200 dark:hover:border-gray-600"
		}
      `}
			onMouseEnter={() => setIsHovered(true)}
			onMouseLeave={() => setIsHovered(false)}
		>
			<div className="flex justify-between items-start mb-4">
				<div>
					<h3 className="text-xl font-medium text-gray-900 dark:text-white">
						{model.name}
					</h3>
					<p className="text-sm text-gray-600 dark:text-gray-300 mt-1">
						{model.fullName}
					</p>
				</div>
			</div>

			<div className="mb-4">
				<div className="inline-block px-3 py-1 rounded-full text-xs font-medium bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200 mb-2">
					{model.type}
				</div>
				<div
					className="inline-block px-3 py-1 rounded-full text-xs font-medium mb-2 ml-2"
					style={{
						backgroundColor: `${model.color}20`,
						color: model.color,
					}}
				>
					{model.purpose}
				</div>
			</div>

			<p className="text-sm text-gray-600 dark:text-gray-300 mb-5">
				{model.description}
			</p>

			<div className="space-y-3">
				{Object.entries(model.metrics).map(([key, value]) => (
					<div key={key}>
						<div className="flex justify-between text-xs mb-1">
							<span className="text-gray-600 dark:text-gray-300">
								{key.charAt(0).toUpperCase() + key.slice(1)}
							</span>
							<span className="font-medium text-gray-900 dark:text-white">
								{value}%
							</span>
						</div>
						<CustomProgress value={value} color={model.color} />
					</div>
				))}
			</div>
		</motion.div>
	);
};

export default ModelCard;
