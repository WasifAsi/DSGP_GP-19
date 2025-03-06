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

          {/* Quick Links */}
          <div className="space-y-6">
            <h3 className="font-display font-medium text-xl text-shoreline-dark dark:text-white">
              Quick Links
            </h3>
            <ul className="space-y-3">
              {[
                { name: 'Home', path: '/' },
                { name: 'Upload & Analysis', path: '/upload' },
                { name: 'About Us', path: '/about' },
                { name: 'Documentation', path: '/docs' }
              ].map((link) => (
                <motion.li
                  key={link.path}
                  whileHover={{ x: 5 }}
                  transition={{ duration: 0.2 }}
                >
                  <Link
                    to={link.path}
                    className="text-shoreline-text dark:text-gray-300 hover:text-shoreline-blue 
                             dark:hover:text-shoreline-blue transition-colors text-sm inline-flex 
                             items-center group"
                  >
                    <span className="h-1 w-4 bg-shoreline-blue/50 rounded mr-2 transform scale-x-0 
                                 group-hover:scale-x-100 transition-transform duration-300" />
                    {link.name}
                  </Link>
                </motion.li>
              ))}
            </ul>
          </div>

          {/* Contact Information */}
          <div className="space-y-6">
            <h3 className="font-display font-medium text-xl text-shoreline-dark dark:text-white">
              Get in Touch
            </h3>
            <address className="not-italic space-y-4">
              {[
                {
                  icon: <MapPin size={20} />,
                  content: "57, Ramakrishna Road, Colombo 06, Sri Lanka",
                  href: "https://goo.gl/maps"
                },
                {
                  icon: <Mail size={20} />,
                  content: "contact@shorelineanalysis.org",
                  href: "mailto:contact@shorelineanalysis.org"
                },
                {
                  icon: <Phone size={20} />,
                  content: "+94 11 234 5678",
                  href: "tel:+94112345678"
                }
              ].map((item, index) => (
                <motion.a
                  key={index}
                  href={item.href}
                  whileHover={{ x: 5 }}
                  className="flex items-center space-x-3 text-shoreline-text dark:text-gray-300 
                           hover:text-shoreline-blue dark:hover:text-shoreline-blue transition-colors group"
                >
                  <span className="text-shoreline-blue group-hover:scale-110 transition-transform duration-300">
                    {item.icon}
                  </span>
                  <span className="text-sm">{item.content}</span>
                </motion.a>
              ))}
            </address>
          </div>
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