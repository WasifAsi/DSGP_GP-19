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
