# Environment Variables

Most environment variables are self-explanatory and are usually used to set an option at runtime.

Example: `GPU=1 DEBUG=4 python3 -m pytest`

The columns are:

* Variable
* Possible Value(s)
* Description.

A `#` means that the variable can take any integer value.

### Global Environment Variables

Global Environment Variables control core tinygrad behavior and are found [here](/docs/environment_variables/global_variables.md).


### File-Specific Environment Variables

File-Specific Environment Variables control the behavior of a specific tinygrad file. They are rarely used, but can be found [here](/docs/environment_variables/file_specific_variables.md).