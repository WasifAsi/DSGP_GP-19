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
						<motion.div
							initial={{ opacity: 0, y: 40 }}
							animate={{ opacity: 1, y: 0 }}
							transition={{ duration: 0.8, delay: 0.3 }}
							className="w-full max-w-5xl relative"
						>
							<div className="absolute -inset-0.5 bg-gradient-to-r from-shoreline-blue/30 to-shoreline-light-blue/30 rounded-2xl blur" />
							<motion.div
								initial={{ opacity: 0, scale: 0.95 }}
								animate={{ opacity: 1, scale: 1 }}
								transition={{ duration: 0.6 }}
								className="relative rounded-2xl overflow-hidden shadow-2xl"
							>
								<img
									src="sea.avif"
									alt="Sri Lankan coastal region"
									className="w-full h-auto rounded-2xl transform hover:scale-105 transition-transform duration-700"
								/>
								<div className="absolute inset-0 bg-gradient-to-b from-transparent to-shoreline-dark/10 pointer-events-none" />
							</motion.div>
						</motion.div>
					</div>
				</div>
			</section>	

			{/* Understanding Coastal Dynamics */}
			<section className="py-16 bg-shoreline-light-blue/20 dark:bg-shoreline-dark/50">
				<div className="container mx-auto px-4">
					<SectionHeading
						eyebrow="Research Focus"
						title={
							<>
								Understanding{" "}
								<span className="text-shoreline-blue">
									Coastal Dynamics
								</span>{" "}
								in Sri Lanka
							</>
						}
						subtitle="Our research focuses on identifying and assessing shoreline changes across critical coastal regions of Sri Lanka to support sustainable environmental management."
						centered={true}
					/>

					<div className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-12">
						<div className="bg-white dark:bg-shoreline-dark/30 p-6 rounded-xl shadow-md">
							<h3 className="text-xl font-medium text-shoreline-dark dark:text-white mb-4">
								The Problem
							</h3>
							<p className="text-shoreline-text dark:text-gray-300">
								Sri Lanka's coastlines are experiencing rapid
								changes due to natural processes, human
								development, and climate-driven forces.
								Understanding these changes is critical for
								effective conservation and management.
							</p>
						</div>

						<div className="bg-white dark:bg-shoreline-dark/30 p-6 rounded-xl shadow-md">
							<h3 className="text-xl font-medium text-shoreline-dark dark:text-white mb-4">
								Our Approach
							</h3>
							<p className="text-shoreline-text dark:text-gray-300">
								By combining satellite imagery analysis with
								machine learning, we're able to identify changes
								with unprecedented accuracy. Our algorithms
								detect shoreline shifts with exceptional
								precision.
							</p>
						</div>

						<div className="bg-white dark:bg-shoreline-dark/30 p-6 rounded-xl shadow-md">
							<h3 className="text-xl font-medium text-shoreline-dark dark:text-white mb-4">
								Impact & Vision
							</h3>
							<p className="text-shoreline-text dark:text-gray-300">
								Our goal is to develop a comprehensive
								understanding of coastal dynamics, provide
								essential data for management decisions, promote
								sustainable conservation, and preserve Sri
								Lanka's valuable coastlines.
							</p>
						</div>
					</div>
				</div>
			</section>

			{/* Features Section */}
			<section className="relative py-24">
				{/* Background Elements */}
				<div className="absolute inset-0 bg-gradient-to-b from-transparent via-shoreline-light-blue/5 to-transparent" />
				<div className="absolute inset-0 bg-grid-pattern opacity-5" />

				<div className="container mx-auto px-4 relative">
					<motion.div
						initial={{ opacity: 0, y: 20 }}
						whileInView={{ opacity: 1, y: 0 }}
						transition={{ duration: 0.6 }}
						viewport={{ once: true }}
					>
						<SectionHeading
							eyebrow="Technologies"
							title={
								<>
									Advanced Tools for{" "}
									<span className="text-shoreline-blue">
										Shoreline Analysis
									</span>
								</>
							}
							subtitle="Our project integrates cutting-edge technologies to create an effective monitoring tool for Sri Lanka's changing coastlines."
							centered={true}
						/>
					</motion.div>
				</div>
			</section>
						
			
		</div>
	);
};

export default Index;
