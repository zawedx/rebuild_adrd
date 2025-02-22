import re
import json

def convert_json_format(input_json):
    def determine_type(descriptor):
        """Determine the type based on the descriptor content."""
        if '\n' not in descriptor:
            return "numerical"
        elif re.search(r'\n\d+ -', descriptor):
            options = re.findall(r'(\d+) - ([^\n]+)', descriptor)
            return "binary" if len(options) == 2 else "multi"
        return "text"

    def parse_descriptor(descriptor):
        """Convert descriptor into the desired format."""
        if '\n' not in descriptor:
            return descriptor
        options = re.findall(r'(\d+) - ([^\n]+)', descriptor)
        base_description = descriptor.split('\n')[0]
        return {key: f"{base_description}\n{value}" for key, value in options}

    output_json = {}

    for key, value in input_json.items():
        descriptor = value.get("Descriptor", "")
        field_type = determine_type(descriptor)

        output_json[key] = {
            "type": field_type,
            "Descriptor": parse_descriptor(descriptor) if field_type in ["binary", "multi"] else descriptor
        }

    return output_json

# # 示例输入JSON数据
# input_json = {
#     "CITY": {
#         "Descriptor": "Ability to correctly identify the current city: \n0 - Stated city is incorrect\n1 - Stated city is correct"
#     },
#     "MMSCORE": {
#         "Descriptor": "Total score from the Mini-Mental State Examination (MMSE), which assesses cognitive function."
#     },
#     "PTPLANG": {
#         "Descriptor": "Primary language of the participant: \n1 - English\n2 - Spanish\n3 - Other languages"
#     },
#     "PTMARRY": {
#         "Descriptor": "Current marital status of the participant: \n1 - Married\n2 - Widowed\n3 - Divorced\n4 - Never married\n5 - Unknown\n6 - Domestic partnership"
#     }
# }

# # 转换JSON数据
# output_json = convert_json_format(input_json)

# # 打印结果
# print(json.dumps(output_json, indent=4))

with open("adni_descriptor2.json", "r") as f:
    input_json = json.load(f)

output_json = convert_json_format(input_json)

with open("adni_descriptor3.json", "w") as f:
    json.dump(output_json, f, indent=4)