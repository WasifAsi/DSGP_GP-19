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

		</div>
	);
};
export default About;
