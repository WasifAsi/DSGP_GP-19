import { useEffect } from "react";
import { motion } from "framer-motion";
import SectionHeading from "../components/SectionHeading";
import AccordionItem from "../components/AccordionItem";
import ModelCard from "../components/ModelCard";
import ModelDiagram from "../components/ModelDiagram";

// Updated model information for U-Net, SegNet, DeepLab v3, and FCN-8
const models = [
	{
		id: "unet",
		name: "U-Net",
		fullName: "U-Net Architecture",
		type: "Semantic Segmentation",
		purpose: "Shoreline Boundary Extraction",
		description:
			"The U-Net architecture excels at pixel-level segmentation. It uses skip connections between encoding and decoding layers, enabling precise localization of shorelines in high-resolution satellite images.",
		metrics: {
			accuracy: 93,
			precision: 94,
			recall: 91,
			f1: 92.5,
		},
		color: "#0092ca",
	},
	{
		id: "segnet",
		name: "SegNet",
		fullName: "SegNet Architecture",
		type: "Semantic Segmentation",
		purpose: "Boundary Delineation",
		description:
			"SegNet is known for its efficient encoder-decoder design. It captures critical shoreline features from large-scale imagery while maintaining lower memory usage.",
		metrics: {
			accuracy: 90,
			precision: 88,
			recall: 89,
			f1: 88.5,
		},
		color: "#7E69AB",
	},
	{
		id: "deeplabv3",
		name: "DeepLab v3",
		fullName: "DeepLab v3",
		type: "Semantic Segmentation",
		purpose: "Robust Feature Extraction",
		description:
			"DeepLab v3 leverages atrous convolution to capture multi-scale context. It’s particularly strong in handling complex shoreline topographies with minimal loss of detail.",
		metrics: {
			accuracy: 92,
			precision: 91,
			recall: 90,
			f1: 90.5,
		},
		color: "#F97316",
	},
	{
		id: "fcn8",
		name: "FCN-8",
		fullName: "Fully Convolutional Network (8s)",
		type: "Semantic Segmentation",
		purpose: "High-Level Pixel Classification",
		description:
			"FCN-8 is a classic fully convolutional network that predicts pixel-level classes using skip layers. It offers a good balance between speed and accuracy for shoreline mapping.",
		metrics: {
			accuracy: 89,
			precision: 87,
			recall: 88,
			f1: 87.5,
		},
		color: "#34D399",
	},
];

const ModelInsightsPage = () => {
	useEffect(() => {
		window.scrollTo(0, 0);
	}, []);

	return (
		<div className="pt-32 pb-24">
			<div className="container mx-auto px-4">
				{/* Hero Section */}
				<section className="container mx-auto px-4 mb-16">
					<div className="flex flex-col md:flex-row items-center">
						<div className="md:w-1/2 mb-8 md:mb-0 md:pr-8">
							<motion.div
								initial={{ opacity: 0, x: -20 }}
								animate={{ opacity: 1, x: 0 }}
								transition={{ duration: 0.5, delay: 0.2 }}
							>
								<SectionHeading
									eyebrow="AI Insights"
									title="Powering Shoreline Analysis with Advanced AI"
									subtitle="Discover how our machine learning models process satellite imagery to detect and predict coastal changes with high accuracy."
									centered={false}
								/>
								<motion.p
									initial={{ opacity: 0 }}
									animate={{ opacity: 1 }}
									transition={{ duration: 0.5, delay: 0.4 }}
									className="mt-4 text-shoreline-text dark:text-gray-300"
								>
									Our suite of specialized AI models work
									together to analyze shoreline data, identify
									erosion patterns, and forecast future
									coastal changes – helping researchers and
									policymakers make informed decisions.
								</motion.p>
							</motion.div>
						</div>
						<div className="md:w-1/2">
							<ModelDiagram />
						</div>
					</div>
				</section>

				{/* Models Performance Section */}
				<section className="mt-24">
					<SectionHeading
						eyebrow="Performance"
						title="Model Metrics"
						subtitle="Detailed performance metrics for each AI model in our system"
						centered={true}
					/>

					{/* 2x2 grid on medium screens */}
					<div className="mt-12 grid grid-cols-1 md:grid-cols-2 gap-8">
						{models.map((model) => (
							<ModelCard key={model.id} model={model} />
						))}
					</div>
				</section>

				{/* FAQ Section */}
				<section className="mt-24">
					<SectionHeading
						eyebrow="FAQ"
						title="Common Questions"
						subtitle="Learn more about our AI models and how they work"
						centered={true}
					/>

					<div className="mt-12 max-w-3xl mx-auto">
						<AccordionItem
							question="Why did we choose these specific AI models?"
							answer="Each of these architectures offers distinct strengths for shoreline segmentation. U-Net excels at capturing fine details, SegNet is memory-efficient, DeepLab v3 handles complex features, and FCN-8 provides a streamlined approach for fast and effective segmentation."
							isOpen={false}
						/>
						<AccordionItem
							question="How accurate are the shoreline predictions?"
							answer="Our models achieve an average accuracy above 90% on a variety of test datasets, with U-Net typically performing best for boundary delineation. DeepLab v3 also excels in complex terrain scenarios."
							isOpen={false}
						/>
						<AccordionItem
							question="How often are the models retrained?"
							answer="The models are retrained quarterly using recent satellite imagery and updated ground-truth data, ensuring they remain accurate as coastal conditions evolve."
							isOpen={false}
						/>
						<AccordionItem
							question="Can these models predict erosion rates?"
							answer="By comparing historical and current segmentation results, we can estimate shoreline movement over time. However, an additional temporal model would typically be used in conjunction with these segmenters for long-term erosion forecasting."
							isOpen={false}
						/>
						<AccordionItem
							question="What data sources are used to train these models?"
							answer="We use Sentinel-2 satellite imagery, historical shoreline records, tidal data, and weather patterns to provide comprehensive coverage of coastal conditions for training."
							isOpen={false}
						/>
					</div>
				</section>
			</div>
		</div>
	);
};

export default ModelInsightsPage;
