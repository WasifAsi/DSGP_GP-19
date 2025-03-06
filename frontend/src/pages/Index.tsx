import { useEffect } from "react";
import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import {
	Satellite,
	Brain,
	MapPin,
	ChevronRight,
	Database,
	Check,
} from "lucide-react";
import SectionHeading from "../components/SectionHeading";
import AccordionItem from "../components/AccordionItem";
// import { motion } from "framer-motion";
// import { Link } from "react-router-dom";
// import {
// 	Satellite,
// 	Brain,
// 	MapPin,
// 	ChevronRight,
// 	Database,
// 	Check,
// } from "lucide-react";
// import SectionHeading from "../components/SectionHeading";
// import AccordionItem from "../components/AccordionItem";

const Index = () => {
	useEffect(() => {
		window.scrollTo(0, 0);
	}, []);

	return (
		<div className="pt-8">
			{/* Hero Section */}
			<section className="py-16 md:py-24">
				<div className="container mx-auto px-4">
					<div className="flex flex-col items-center">
						<motion.div
							initial={{ opacity: 0, x: -20 }}
							animate={{ opacity: 1, x: 0 }}
							transition={{ duration: 0.6 }}
							className="text-center max-w-4xl mb-12"
						>
							<span className="inline-block px-3 py-1 rounded-full text-xs font-medium bg-shoreline-light-blue text-shoreline-blue mb-4">
								Coastal Research Study
							</span>
							<h1 className="text-4xl md:text-5xl lg:text-6xl font-display font-medium text-shoreline-dark dark:text-white mb-6 leading-tight">
								Analyzing Shoreline Changes in{" "}
								<span className="text-shoreline-blue">
									Sri Lanka's
								</span>{" "}
								Critical Regions
							</h1>
							<p className="text-lg text-shoreline-text dark:text-gray-300 mb-8 mx-auto max-w-2xl">
								Using satellite imagery and machine learning to
								monitor, analyze, and predict coastal erosion
								patterns for better environmental management.
							</p>
							<div className="flex flex-col sm:flex-row gap-4 justify-center">
								<Link
									to="/upload"
									className="upload-btn text-center"
								>
									Upload Satellite Image
									<ChevronRight
										size={16}
										className="ml-2 inline"
									/>
								</Link>
								<Link
									to="/about"
									className="px-6 py-2 border border-gray-300 dark:border-gray-700 text-shoreline-dark dark:text-white rounded-md transition-all duration-300 hover:border-shoreline-blue hover:text-shoreline-blue text-center"
								>
									Learn More
								</Link>
							</div>
						</motion.div>
					</div>
				</div>
			</section>	
						
			
		</div>
	);
};

export default Index;
