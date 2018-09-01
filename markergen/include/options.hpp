#pragma once

#include <boost/program_options.hpp>
#include <string>
#include <cstdint>

class Options {
public:

    Options(int argc, const char* const argv[]);

    bool showOnlyHelp() const noexcept { return m_showOnlyHelp; }
    uint64_t code() const noexcept { return m_code; }
    int blockWidth() const noexcept { return m_blockWidth; }
    const std::string& filename() const noexcept { return m_filename; }

    friend std::ostream& operator<< (std::ostream& os, const Options& opt);

private:
    boost::program_options::options_description m_description;
    uint64_t m_code;
    int m_blockWidth;
    std::string m_filename;
    bool m_showOnlyHelp;
};
