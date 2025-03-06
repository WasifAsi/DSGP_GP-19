import { useEffect } from "react";
import SectionHeading from "../components/SectionHeading";
// import TeamMember from "../components/TeamMember";
// import ContactForm from "../components/ContactForm";
// import { ExternalLink, ArrowRight, Mail, Phone, MapPin } from "lucide-react";
// import { Link } from "react-router-dom";

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
						</div>

		</div>
	);
};
export default About;
