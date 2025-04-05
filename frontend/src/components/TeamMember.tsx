import { motion } from 'framer-motion';
import { Linkedin, Mail } from 'lucide-react';

interface TeamMemberProps {

  name: string;
  title: string;
  affiliation: string;
  description: string;
  imageSrc: string;
  delay?: number;
}

const TeamMember = ({ name, title, affiliation, description, imageSrc, delay = 0 }: TeamMemberProps) => {

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: delay * 0.1 }}
      viewport={{ once: true, margin: "-100px" }}
      className="glossy-card p-6 dark:bg-gray-800/50 dark:backdrop-blur-xl"
    >
      <div className="flex flex-col items-center text-center">


      <div className="relative w-24 h-24 mb-4 overflow-hidden rounded-full border-2 border-shoreline-blue/10 dark:border-shoreline-blue/20">
        <img
        src={imageSrc}
        alt={name}
        className="w-full h-full object-cover"
        />
      </div>

      <h3 className="text-lg font-medium text-shoreline-dark dark:text-white">{name}</h3>
      <p className="text-sm text-shoreline-blue font-medium mb-1">{title}</p>
      <p className="text-xs text-shoreline-text dark:text-gray-400 mb-3">{affiliation}</p>
      
      <p className="text-sm text-shoreline-text dark:text-gray-300 mb-4">{description}</p>

      <div className="flex space-x-3">
        <a
        href="#"
        className="w-8 h-8 flex items-center justify-center rounded-full bg-shoreline-light-blue/50 text-shoreline-blue hover:bg-shoreline-blue hover:text-white transition-colors dark:bg-shoreline-blue/20"
        >
        <Linkedin size={16} />
        </a>
        <a
        href="#"
        className="w-8 h-8 flex items-center justify-center rounded-full bg-shoreline-light-blue/50 text-shoreline-blue hover:bg-shoreline-blue hover:text-white transition-colors dark:bg-shoreline-blue/20"
        >
        <Mail size={16} />
        </a>
      </div>
      </div>
    </motion.div>
  );
};

export default TeamMember;
