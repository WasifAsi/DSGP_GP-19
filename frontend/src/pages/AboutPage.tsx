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

				<div className="grid grid-cols-1 lg:grid-cols-2 gap-16 mt-12">
					<div>
						<h3 className="text-xl font-medium text-shoreline-dark dark:text-white mb-6">
							Get in Touch
						</h3>
						<p className="text-shoreline-text dark:text-gray-300 mb-8">
							Have questions about our research or interested in
							collaboration? Reach out to our team through this
							form or the contact details provided.
						</p>

						<div className="space-y-6">
							<div className="flex items-start">
								<div className="w-10 h-10 rounded-full bg-shoreline-light-blue/50 flex items-center justify-center mr-4 flex-shrink-0">
									<Mail className="h-5 w-5 text-shoreline-blue" />
								</div>
								<div>
									<h4 className="font-medium text-shoreline-dark dark:text-white mb-1">
										Email
									</h4>
									<a
										href="mailto:contact@shorelineanalysis.org"
										className="text-shoreline-text dark:text-gray-300 hover:text-shoreline-blue dark:hover:text-shoreline-blue transition-colors"
									>
										contact@shorelineanalysis.org
									</a>
								</div>
							</div>

							<div className="flex items-start">
								<div className="w-10 h-10 rounded-full bg-shoreline-light-blue/50 flex items-center justify-center mr-4 flex-shrink-0">
									<Phone className="h-5 w-5 text-shoreline-blue" />
								</div>
								<div>
									<h4 className="font-medium text-shoreline-dark dark:text-white mb-1">
										Phone
									</h4>
									<a
										href="tel:+94 11 234 5678"
										className="text-shoreline-text dark:text-gray-300 hover:text-shoreline-blue dark:hover:text-shoreline-blue transition-colors"
									>
										+94 11 234 5678
									</a>
								</div>
							</div>
							<div className="flex items-start">
								<div className="w-10 h-10 rounded-full bg-shoreline-light-blue/50 flex items-center justify-center mr-4 flex-shrink-0">
									<MapPin className="h-5 w-5 text-shoreline-blue" />
								</div>
								<div>
									<h4 className="font-medium text-shoreline-dark dark:text-white mb-1">
										Address
									</h4>
									<address className="not-italic text-shoreline-text dark:text-gray-300">
										57, Ramakrishna Road
										<br />
										Colombo 06, Sri Lanka
									</address>
								</div>
							</div>
						</div>
						<div className="mt-8 h-64 rounded-xl overflow-hidden">
							<iframe
								src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d63371.80385596634!2d79.83789389349902!3d6.927079595149932!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x3ae253d10f7a7003%3A0x320b2e4d32d3838d!2sColombo%2C%20Sri%20Lanka!5e0!3m2!1sen!2sus!4v1616418743741!5m2!1sen!2sus"
								width="100%"
								height="100%"
								style={{ border: 0 }}
								allowFullScreen={true}
								loading="lazy"
								title="Map of Colombo, Sri Lanka"
							></iframe>
						</div>
					</div>




				</div>		

				

		</div>
	);
};
export default About;
