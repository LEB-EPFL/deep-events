... mermaid::
    flowChart
        origin_folder
        destination_folder (default in settings)

    origin_folder - find all subfolders with tifs -> subfolder_list
    subfolder_list - for each one get all the info necessary

    Overall: Look in subfolder for info, if not there, look in name
    of subfolder. If not there, move up look in .yaml files, if not,
    look in name of superfolder. Repeat until reaching origin_folder.

    Duplicate files: Check in subfolder for instruction on which one to use
    if no info, take newest one. Just put info with file ending into db.yaml
    file.

    If no info: if no info for an essential entry can be found, ask for it.
    If not essential output warning with subfolder location.
    -> Do this check before doing any computational intensive things

    -> No manual_entries file. Put this info into the folders
    -> For keys, allow groups that will be named as the first entry of the groups