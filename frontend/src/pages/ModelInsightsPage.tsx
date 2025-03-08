import { useEffect } from "react";
import { motion } from "framer-motion";
import SectionHeading from "../components/SectionHeading";
import AccordionItem from "../components/AccordionItem";
import ModelCard from "../components/ModelCard";
import ModelDiagram from "../components/ModelDiagram";
import ModelTable from "../components/ModelTable";

// Import images from assets
import model1Image from "../assets/images/model1.png";
import model2Image from "../assets/images/model2.png";
import model3Image from "../assets/images/model3.png";
import model4Image from "../assets/images/model4.png";

// Define TypeScript interface for Model
interface Model {
	id: string;
	name: string;
	fullNameOverview: string;
	fullNamePerformance: string;
	type: string;
	purpose: string;
	descriptionOverview: string;
	descriptionPerformance: string;
	metrics: {
		accuracy: number;
		precision: number;
		recall: number;
		f1: number;
	};
	color: string;
	image: string;
}

// Mapping Model IDs to their images
const modelImages: Record<string, string> = {
	model1: model1Image,
	model2: model2Image,
	model3: model3Image,
	model4: model4Image,
};

// Models Data (Strictly typed)
const models: Model[] = [
	{
		id: "model1",
		name: "U-Net",
		fullNameOverview: "U-Net Model Architecture",
		fullNamePerformance: "U-Net Model Performance Overview",
		type: "Semantic Segmentation",
		purpose: "Coastal Boundary Detection",
		descriptionOverview:
			"U-Net is a convolutional neural network originally developed for biomedical image segmentation. It features a symmetric encoder-decoder architecture with skip connections. These skip connections merge high-resolution features from the encoder directly to the decoder, improving segmentation detail, especially for small objects.",
		descriptionPerformance:
			"Highly effective for segmentation tasks involving detailed boundaries and smaller datasets. Known for excellent accuracy in medical image segmentation. It often achieves good results with fewer training samples.",
		metrics: {
			accuracy: 93,
			precision: 94,
			recall: 91,
			f1: 92.5,
		},
		color: "#0092ca",
		image: modelImages["model1"],
	},
	{
		id: "model2",
		name: "SegNet",
		fullNameOverview: "SegNet Model Architecture",
		fullNamePerformance: "SegNet Model Performance Overview",
		type: "Semantic Segmentation",
		purpose: "Efficient Shoreline Mapping",
		descriptionOverview:
			"SegNet employs an encoder-decoder structure similar to U-Net but differs in how features are upsampled. Instead of skip connections, SegNet stores pooling indices from the encoder for use in the decoder. This approach reduces memory usage but can lose some detail compared to methods with explicit skip connections.",
		descriptionPerformance:
			"Generally computationally efficient and lightweight, suitable for real-time segmentation tasks. It might have slightly lower segmentation accuracy than U-Net and DeepLab, especially in fine-detail preservation.",
		metrics: {
			accuracy: 90,
			precision: 88,
			recall: 89,
			f1: 88.5,
		},
		color: "#7E69AB",
		image: modelImages["model2"],
	},
	{
		id: "model3",
		name: "DeepLab v3",
		fullNameOverview: "DeepLab v3 Model Architecture",
		fullNamePerformance: "DeepLab v3 Model Performance Overview",
		type: "Semantic Segmentation",
		purpose: "High-Resolution Shoreline Analysis",
		descriptionOverview:
			"DeepLab v3 integrates atrous (dilated) convolutions and spatial pyramid pooling (ASPP) to capture multi-scale contextual information without significantly increasing computational cost. It typically uses a backbone like ResNet for feature extraction, which enhances accuracy.",
		descriptionPerformance:
			"Excellent segmentation accuracy due to multi-scale context aggregation. Often achieves state-of-the-art results on challenging benchmarks like Cityscapes or Pascal VOC. More resource-intensive than SegNet and U-Net but capable of capturing context-rich details efficiently.",
		metrics: {
			accuracy: 92,
			precision: 91,
			recall: 90,
			f1: 90.5,
		},
		color: "#F97316",
		image: modelImages["model3"],
	},
	{
		id: "model4",
		name: "FCN-8",
		fullNameOverview: "FCN-8 Model Architecture",
		fullNamePerformance: "FCN-8 Model Performance Overview",
		type: "Semantic Segmentation",
		purpose: "Automated Coastal Classification",
		descriptionOverview:
			"FCN-8 transforms standard CNN architectures (e.g., VGG16) into fully convolutional networks to achieve pixel-wise predictions. The 8 refers to the stride size at the finest upsampling step. It combines features from intermediate layers (skip connections) to refine segmentation resolution.",
		descriptionPerformance:
			"Typically offers solid baseline performance for general segmentation tasks. Simpler structure than DeepLab or U-Net, making it less computationally expensive. However, its accuracy and ability to capture fine details are generally surpassed by more advanced models like U-Net and DeepLab v3.",
		metrics: {
			accuracy: 89,
			precision: 87,
			recall: 88,
			f1: 87.5,
		},
		color: "#34D399",
		image: modelImages["model4"],
	},
];

// React Functional Component
const ModelInsightsPage: React.FC = () => {
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
									eyebrow="ML Insights"
									title="Powering Shoreline Analysis with Advanced ML"
									subtitle="Discover how our advanced machine learning models analyze satellite imagery to accurately detect and predict coastal changes. Leveraging state-of-the-art deep learning techniques such as U-Net, SegNet, DeepLab v3, and FCN-8, our models efficiently identify patterns of erosion, sediment movement, flooding, and ecological shifts."
									centered={false}
								/>
								<motion.p
									initial={{ opacity: 0 }}
									animate={{ opacity: 1 }}
									transition={{ duration: 0.5, delay: 0.4 }}
									className="mt-4 text-shoreline-text dark:text-gray-300"
								>
									Starting with high-quality satellite data, we preprocess and analyze images to
									reveal crucial insights into coastal dynamics. Our predictive capabilities enable
									proactive decision-making, helping stakeholders manage coastal resources, plan
									infrastructure, and improve ecological conservation efforts.
								</motion.p>
							</motion.div>
						</div>
						<div className="md:w-1/2">
							<ModelDiagram />
						</div>
					</div>
				</section>

				{/* Model Table Section */}
				<ModelTable models={models} />

				{/* Models Performance Section */}
				<section className="mt-24">
					<SectionHeading
						eyebrow="Performance"
						title="Model Metrics"
						subtitle="Detailed performance metrics for each ML model in our system"
						centered={true}
					/>
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
					subtitle="Learn more about our ML models and how they work"
					centered={true}
				/>
				<div className="mt-12 max-w-3xl mx-auto">
					{[
					{
						question: "Why did we choose these specific ML models?",
						answer:
						"We selected these models based on their performance in geospatial analysis, ability to handle time-series data, and their robustness in predicting shoreline changes.",
					},
					{
						question: "How accurate are the shoreline predictions?",
						answer:
						"The models achieve a high degree of accuracy, typically above 85%, depending on the quality and resolution of the input data.",
					},
					{
						question: "How often are the models retrained?",
						answer:
						"The models are retrained quarterly using updated satellite imagery and real-time coastal data to ensure the predictions remain accurate.",
					},
					{
						question: "Can these models predict erosion rates?",
						answer:
						"Yes, our models analyze historical shoreline data to predict erosion rates and highlight high-risk coastal areas.",
					},
					{
						question: "What data sources are used to train these models?",
						answer:
						"The models are trained using satellite imagery, LiDAR scans, tide gauge data, and historical shoreline movement records.",
					},
					].map((faq, index) => (
					<AccordionItem key={index} question={faq.question} answer={faq.answer} isOpen={false} />
					))}
				</div>
				</section>
			</div>
		</div>
	);
};

export default ModelInsightsPage;
