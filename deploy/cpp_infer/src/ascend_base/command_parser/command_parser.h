#ifndef ASCEND_BASE_COMMAND_PARSER_H
#define ASCEND_BASE_COMMAND_PARSER_H

#include <string>
#include <map>
#include <vector>
#include "status_code/status_code.h"

// Command parser class
class CommandParser {
public:
    CommandParser();

    // Construct a new Command Parser object according to the argument
    CommandParser(int argc, const char **argv);

    ~CommandParser() = default;

    // Add options into the map
    void AddOption(const std::string &option, const std::string &defaults = "", const std::string &message = "");

    // Parse the input arguments
    void ParseArgs(int argc, const char **argv);

    // Get the option string value from parser
    const std::string &GetStringOption(const std::string &option);

    // Get the int value by option
    int GetIntOption(const std::string &option);

    uint32_t GetUint32Option(const std::string &option);

    // Get the int value by option
    float GetFloatOption(const std::string &option);

    // Get the double option
    double GetDoubleOption(const std::string &option);

    // Get the bool option
    bool GetBoolOption(const std::string &option);

    // Get int vector
    Status GetVectorUint32Value(const std::string &option, std::vector<uint32_t> &vector);
private:
    std::map<std::string, std::pair<std::string, std::string>> commands_;

    // Show the usage of app, then exit
    void ShowUsage() const;

    bool IsInteger(std::string &str) const;

    bool IsDecimal(std::string &str) const;
};

#endif
