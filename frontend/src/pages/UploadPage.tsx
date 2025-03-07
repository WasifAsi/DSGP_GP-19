import { useEffect, useState } from "react";
import SectionHeading from "../components/SectionHeading";
import { motion } from "framer-motion";
import { toast } from "sonner";
import UploadArea from "../components/UploadArea";
import { ArrowRight, Layers, Activity, Map } from "lucide-react";

const Upload = () => {
	const [isUploading, setIsUploading] = useState(false);
	const [uploadProgress, setUploadProgress] = useState(0);
	const [analysisStep, setAnalysisStep] = useState(0);
	const [analysisComplete, setAnalysisComplete] = useState(false);

	useEffect(() => {
		window.scrollTo(0, 0);
	}, []);

	const handleUpload = (file: File) => {
		// Validate file
		if (!file) {
			toast.error("Please select a file to upload");
			return;
		}

		// Validate file type (assuming we accept image files)
		const allowedTypes = ["image/jpeg", "image/png", "image/tiff"];
		if (!allowedTypes.includes(file.type)) {
			toast.error(
				"Please upload a valid image file (JPEG, PNG, or TIFF)"
			);
			return;
		}

		// Validate file size (e.g., max 50MB)
		const maxSize = 50 * 1024 * 1024; // 50MB in bytes
		if (file.size > maxSize) {
			toast.error("File size must be less than 50MB");
			return;
		}

		setIsUploading(true);
		setUploadProgress(0);
		setAnalysisStep(0);
		setAnalysisComplete(false);

		// Simulate upload progress
		const interval = setInterval(() => {
			setUploadProgress((prev) => {
				if (prev >= 100) {
					clearInterval(interval);
					startAnalysis();
					return 100;
				}
				return prev + 5;
			});
		}, 200);
	};

	const startAnalysis = () => {
		setIsUploading(false);

		// Simulate analysis steps
		setTimeout(() => setAnalysisStep(1), 1000);
		setTimeout(() => setAnalysisStep(2), 3000);
		setTimeout(() => setAnalysisStep(3), 5000);
		setTimeout(() => {
			setAnalysisStep(4);
			setAnalysisComplete(true);
			toast.success("Analysis completed successfully!", {
				description: "Your shoreline analysis results are ready.",
			});
		}, 7000);
	};

	const analysisSteps = [
		{
			name: "Preprocessing image",
			description:
				"Applying radiometric calibration and atmospheric correction",
		},
		{
			name: "Detecting shoreline",
			description:
				"Applying machine learning algorithms to identify water-land boundaries",
		},
		{
			name: "Measuring changes",
			description: "Comparing with historical shoreline positions",
		},
		{
			name: "Generating report",
			description: "Compiling analysis results and visualizations",
		},
	];

	return (
		<div className="pt-32 pb-24">
			<div className="container mx-auto px-4">
				<SectionHeading
						eyebrow="Upload & Analyze"
						title="Shoreline Change Analysis"
						subtitle="Upload satellite images of Sri Lankan coastal regions to analyze shoreline changes and erosion patterns. Our system will automatically detect shorelines and calculate rates of change."
				/>
			
				<div className="mt-16 max-w-4xl mx-auto">
					<UploadArea
						onUpload={handleUpload}
						isUploading={isUploading}
						uploadProgress={uploadProgress}
					/>
				</div>

				{analysisStep > 0 && (
					<motion.div
						initial={{ opacity: 0, y: 20 }}
						animate={{ opacity: 1, y: 0 }}
						transition={{ duration: 0.5 }}
						className="mt-16 max-w-3xl mx-auto"
					>
						<div className="bg-white dark:bg-gray-800/30 rounded-xl shadow-lg p-8">
							<h3 className="text-xl font-medium text-shoreline-dark dark:text-white mb-6">
								Analysis Progress
							</h3>

							<div className="space-y-6">
								{analysisSteps.map((step, index) => (
									<div
										key={index}
										className="flex items-start"
									>
										<div
											className={`w-8 h-8 rounded-full flex items-center justify-center mr-4 flex-shrink-0 ${
												analysisStep > index
													? "bg-green-100 text-green-600 dark:bg-green-900/30 dark:text-green-400"
													: analysisStep === index
													? "bg-shoreline-light-blue text-shoreline-blue"
													: "bg-gray-100 text-gray-400 dark:bg-gray-700 dark:text-gray-500"
											}`}
										>
											{analysisStep > index ? (
												<svg
													className="w-5 h-5"
													fill="none"
													stroke="currentColor"
													viewBox="0 0 24 24"
													xmlns="http://www.w3.org/2000/svg"
												>
													<path
														strokeLinecap="round"
														strokeLinejoin="round"
														strokeWidth="2"
														d="M5 13l4 4L19 7"
													></path>
												</svg>
											) : (
												<span>{index + 1}</span>
											)}
										</div>

										<div className="flex-1">
											<div className="flex justify-between">
												<h4
													className={`font-medium ${
														analysisStep >= index
															? "text-shoreline-dark dark:text-white"
															: "text-gray-400 dark:text-gray-500"
													}`}
												>
													{step.name}
												</h4>
												{analysisStep === index &&
													!analysisComplete && (
														<div className="animate-pulse text-shoreline-blue text-sm">
															Processing...
														</div>
													)}
											</div>
											<p
												className={`text-sm ${
													analysisStep >= index
														? "text-shoreline-text dark:text-gray-300"
														: "text-gray-400 dark:text-gray-500"
												}`}
											>
												{step.description}
											</p>
										</div>
									</div>
								))}
							</div>

							{/* todo  */}

						</div>
					</motion.div>
				)}

				{/* todo  */}

			</div>
		</div>
	);
};

export default Upload;
