import { motion } from 'framer-motion';

interface SectionHeadingProps {
  eyebrow: string;
  title: string | React.ReactNode;
  subtitle?: string;
  centered?: boolean;
}

const SectionHeading = ({ eyebrow, title, subtitle, centered = true }: SectionHeadingProps) => {
  return (
    <div className={`mb-12 ${centered ? 'text-center' : 'text-left'}`}>
      <motion.span
        initial={{ opacity: 0, y: 10 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        viewport={{ once: true, margin: "-100px" }}
        className="inline-block px-3 py-1 rounded-full text-xs font-medium bg-shoreline-light-blue text-shoreline-blue mb-3"
      >
        {eyebrow}
      </motion.span>
      
      <motion.h2
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.1 }}
        viewport={{ once: true, margin: "-100px" }}
        className="text-3xl md:text-4xl font-display font-medium text-shoreline-dark dark:text-white mb-4"
      >
        {title}
      </motion.h2>
      
      {subtitle && (
        <motion.p
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.2 }}
          viewport={{ once: true, margin: "-100px" }}
          className="text-shoreline-text dark:text-gray-300 max-w-2xl mx-auto"
        >
          {subtitle}
        </motion.p>
      )}
    </div>
  );
};

export default SectionHeading;