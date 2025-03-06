import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { Mail, Phone, MapPin, Github, Twitter, Linkedin } from 'lucide-react';

const Footer = () => {
  const footerAnimation = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0 }
  };

  return (
    <footer className="relative bg-gradient-to-b from-shoreline-light to-white dark:from-shoreline-dark/95 dark:to-gray-900/95 pt-16 pb-8 overflow-hidden">
      {/* Background Pattern */}
      <div className="absolute inset-0 bg-grid-pattern opacity-5" />
      
      <div className="container mx-auto px-4 relative">
        <motion.div 
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.2 }}
            variants={footerAnimation}
            className="grid grid-cols-1 md:grid-cols-3 gap-12 pb-12"
          >

          {/* to do */}

        </motion.div>

        {/* Footer Bottom */}
        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.4 }}
          className="border-t border-gray-200 dark:border-gray-700/50 pt-8 mt-8"
        >
          <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
            <p className="text-sm text-shoreline-text dark:text-gray-400">
              © {new Date().getFullYear()} Shoreline Analysis. All rights reserved.
            </p>
            <div className="flex space-x-8">
              {['Privacy Policy', 'Terms of Service'].map((item) => (
                <Link
                  key={item}
                  to={`/${item.toLowerCase().replace(/\s+/g, '-')}`}
                  className="text-sm text-shoreline-text dark:text-gray-400 hover:text-shoreline-blue 
                           dark:hover:text-shoreline-blue transition-colors"
                >
                  {item}
                </Link>
              ))}
            </div>
          </div>
        </motion.div>
      </div>
    </footer>
  );
};

export default Footer;