#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <regex>

#include "fontGlyph.cuh"

class fntParser
{
public:
	std::string face, charSet;
	int size = 0, bold = 0, italic = 0, unicode = 0, stretchH = 0, smooth = 0, aa = 0;
	std::vector<int> padding, spacing;
	int lineHeight = 0, base = 0, scaleW = 0, scaleH = 0, pages = 0, packed = 0;

	std::vector<fontGlyph> glyphs;

	fntParser(std::wstring fileName)
	{
		std::ifstream file(fileName);
		std::string line;
		while (file.is_open() && std::getline(file, line))
		{
			if (line.rfind("info ", 0) == 0)
			{
				parseInfoLine(line);
			}
			if (line.rfind("common ", 0) == 0)
			{
				parseCommonLine(line);
			}
			if (line.rfind("char ", 0) == 0)
			{
				parseCharLine(line);
			}
		}

		file.close();
	}

	bool getIntParameter(std::string line, std::string paramName, int& value)
	{
		auto paramStr = paramName + "=";
		auto idx = line.find(paramStr, 0);
		if (idx != std::string::npos)
		{
			auto startIdx = idx + paramStr.length();
			auto endIdx = line.find(" ", idx);
			if (endIdx != std::string::npos)
			{
				try
				{
					value = std::stoi(line.substr(startIdx, endIdx - startIdx));
					return true;
				}
				catch (const std::exception&)
				{
					return false;
				}
			}
		}

		return false;
	}

	bool getStringParameter(std::string line, std::string paramName, std::string& value)
	{
		auto paramStr = paramName + "=\"";
		auto idx = line.find(paramStr, 0);
		if (idx != std::string::npos)
		{
			auto startIdx = idx + paramStr.length();
			auto endIdx = line.find("\" ", idx);
			if (endIdx != std::string::npos)
			{
				try
				{
					value = line.substr(startIdx, endIdx - startIdx);
					return true;
				}
				catch (const std::exception&)
				{
					return false;
				}
			}
		}

		return false;
	}

	void parseInfoLine(std::string line)
	{
		getStringParameter(line, "face", face);
		getStringParameter(line, "charset", charSet);

		getIntParameter(line, "size", size);
		getIntParameter(line, "bold", bold);
		getIntParameter(line, "italic", italic);
		getIntParameter(line, "unicode", unicode);
		getIntParameter(line, "stretchH", stretchH);
		getIntParameter(line, "smooth", smooth);
		getIntParameter(line, "aa", aa);
	}

	void parseCommonLine(std::string line)
	{
		getIntParameter(line, "lineHeight", lineHeight);
		getIntParameter(line, "base", base);
		getIntParameter(line, "scaleW", scaleW);
		getIntParameter(line, "scaleH", scaleH);
		getIntParameter(line, "pages", pages);
		getIntParameter(line, "packed", packed);
	}

	void parseCharLine(std::string line)
	{
		int id, x, y, width, height, xOffset, yOffset, xAdvance;

		if (!getIntParameter(line, "id", id)) return;
		if (!getIntParameter(line, "x", x)) return;
		if (!getIntParameter(line, "y", y)) return;
		if (!getIntParameter(line, "width", width)) return;
		if (!getIntParameter(line, "height", height)) return;
		if (!getIntParameter(line, "xoffset", xOffset)) return;
		if (!getIntParameter(line, "yoffset", yOffset)) return;
		if (!getIntParameter(line, "xadvance", xAdvance)) return;

		glyphs.push_back(fontGlyph(id, x, y, width, height, xOffset, yOffset, xAdvance));
	}
};