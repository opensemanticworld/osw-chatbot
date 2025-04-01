from pydantic import BaseModel, Field


class GetFileHeaderInput(BaseModel):
    file_path: str = Field(description="The path to the file to get the header from.")
    n_lines: int = Field(
        default=10, description="The number of lines to read from the file."
    )


def get_file_header(inp: GetFileHeaderInput):
    """a function that reads the header of a file and returns it as text"""
    if inp.file_path.endswith(".txt") or inp.file_path.endswith(".csv"):
        with open(inp.file_path, "r") as file:
            lines = [next(file) for _ in range(10)]
        return "".join(lines)


if __name__ == "__main__":
    header = get_file_header(GetFileHeaderInput(file_path="test.csv"))
    print(header)
