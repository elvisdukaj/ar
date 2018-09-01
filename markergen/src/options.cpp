#include "options.hpp"


namespace po = boost::program_options;

Options::Options(int argc, const char* const argv[])
{
    m_description.add_options()
            ("help,h", "Help message")
            ("code,c", po::value<uint64_t>(&m_code)->default_value(uint64_t(0)), "code to generate")
            ("block-size,b", po::value<int>(&m_blockWidth)->default_value(10), "width in pixel of the block")
            ("filename,f", po::value<std::string>(&m_filename));

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, m_description), vm);
    po::notify(vm);

    m_showOnlyHelp = vm.count("help") != 0;
}

std::ostream& operator<< (std::ostream& os, const Options& opt)
{
    return os << opt.m_description;
}
