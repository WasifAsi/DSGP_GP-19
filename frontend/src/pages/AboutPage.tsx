import { useEffect } from "react";
import SectionHeading from "../components/SectionHeading";
import TeamMember from "../components/TeamMember";
import ContactForm from "../components/ContactForm";
import { ExternalLink, ArrowRight, Mail, Phone, MapPin } from "lucide-react";
import { Link } from "react-router-dom";

const About = () => {
	useEffect(() => {
		window.scrollTo(0, 0);
	}, []);
	return (
		<div className="pt-32 pb-24">
			<div className="container mx-auto px-4"></div>
			<SectionHeading
					eyebrow="About Us"
					title="The Team Behind Shoreline Analysis"
					subtitle="Our interdisciplinary team brings together expertise from the Informatics Institute of Technology and Robert Gordon University to address critical coastal management challenges in Sri Lanka."
				/>
			
			<div className="section-divider" />
			<div className="mt-16 grid grid-cols-1 md:grid-cols-2 gap-12 items-center"></div>
			  <div>
					<h3 className="text-2xl font-medium text-shoreline-dark dark:text-white mb-4">
							Why We Started
					</h3>
					<p className="text-shoreline-text dark:text-gray-300 mb-6">
							Our project began in response to the growing
							challenges facing Sri Lanka's coastlines. With
							rising sea levels and increasing coastal
							development, we recognized the need for advanced
							monitoring tools to support evidence-based coastal
							management decisions.
						</p>
						<p className="text-shoreline-text dark:text-gray-300 mb-6">
							By leveraging satellite technology and machine
							learning, we're making it easier for researchers,
							environmental agencies, and policymakers to track
							and respond to coastal changes.
						</p>
						<Link
							to="/upload"
							className="inline-flex items-center text-shoreline-blue hover:text-shoreline-dark transition-colors"
						>
							Try our analysis tools
							<ArrowRight size={16} className="ml-2" />
						</Link>
						</div>
						<div className="relative">
						<img
							src="/coastal-research.jpg"
							alt="Coastal research in action"
							className="rounded-lg shadow-lg"
						/>
						<div className="absolute inset-0 bg-gradient-to-r from-shoreline-blue/20 to-transparent rounded-lg" />
					</div>

				    <div className="section-divider" />

					<SectionHeading
					eyebrow="Our Team"
					title="Meet the Researchers"
					subtitle="Our interdisciplinary team combines expertise in remote sensing, machine learning, environmental science, and coastal engineering."
				    />

                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-8 mt-12">
						<TeamMember
						   name="Member Name"
						   title="Lead Researcher"
						   affiliation="Informatics Institute of Technology"
						   description="Expert in coastal geomorphology with 10+ years of experience mapping Sri Lankan shorelines."
						   imageSrc="/user.jpeg"
						   delay={0}
						/>

                        <TeamMember
						    name="Member Name"
						    title="Machine Learning Engineer"
						    affiliation="Informatics Institute of Technology"
						    description="Specializes in geospatial analysis and applying image processing for environmental applications."
						    imageSrc="/user.jpeg"
						    delay={1}
					    />

                        <TeamMember
						    name="Member Name"
						    title="Software Development Engineer"
						    affiliation="Robert Gordon University"
						    description="Develops algorithms for automated shoreline detection to enhance precision."
						    imageSrc="/user.jpeg"
						    delay={2}
					    />

					    <TeamMember
						    name="Member Name"
						    title="Environmental Scientist"
						    affiliation="Robert Gordon University"
						    description="Studies coastal ecosystems and the environmental impact of shoreline changes."
						    imageSrc="/user.jpeg"
						    delay={3}
					    />
				</div>

				<div className="mt-16 grid grid-cols-1 md:grid-cols-2 gap-8">
					<div className="bg-white dark:bg-shoreline-dark/30 p-6 rounded-xl shadow-md">
						<h3 className="text-lg font-medium text-shoreline-dark dark:text-white mb-2">
							Project Supervision
						</h3>
						<div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-4">
							<div className="p-4">
								<p className="text-shoreline-blue font-medium text-sm">
									Project Supervisor
								</p>
								<h4 className="text-shoreline-dark dark:text-white font-medium mb-1">
									Prof. Kavinda Silva
								</h4>
								<p className="text-sm text-shoreline-text dark:text-gray-400">
									Informatics Institute of Technology
								</p>
							</div>
							<div className="p-4">
								<p className="text-shoreline-blue font-medium text-sm">
									Research Advisor
								</p>
								<h4 className="text-shoreline-dark dark:text-white font-medium mb-1">
									Dr. Emily Anderson
								</h4>
								<p className="text-sm text-shoreline-text dark:text-gray-400">
									Robert Gordon University
								</p>
							</div>
							</div>
					</div>

					<div className="bg-white dark:bg-shoreline-dark/30 p-6 rounded-xl shadow-md">
						<h3 className="text-lg font-medium text-shoreline-dark dark:text-white mb-4">
							Acknowledgements
						</h3>
						<p className="text-shoreline-text dark:text-gray-300 mb-4">
							This research is supported by grants from the
							National Science Foundation of Sri Lanka and the
							Environmental Protection Agency. We also thank the
							Department of Coast Conservation and Coastal
							Resource Management for providing data access and
							field support.
						</p>
						<a
							href="#"
							className="inline-flex items-center text-shoreline-blue hover:underline font-medium text-sm"
						>
							<span>Visit Funding Partners</span>
							<ExternalLink size={14} className="ml-1" />
						</a>
						</div>
				</div>

				<div className="section-divider" />

				<SectionHeading
					eyebrow="Contact"
					title="Get in Touch"
					subtitle="Have questions about our research or interested in collaboration? We'd love to hear from you."
				/>


				</div>		

				

		</div>
	);
};
export default About;
