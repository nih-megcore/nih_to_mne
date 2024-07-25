#! /usr/bin/env python3

import os, sys

# The tags have to be output in this order ...
fiducials0 = [ "Nasion", "Left Ear", "Right Ear" ]
fiducials = [ "Nasion", "Left Ear", "Right Ear", "LPA", "RPA" ]
alt_fid = { "LPA" : "Left Ear", "RPA" : "Right Ear" }
# alt_fid2 = {"left fiducial":"Left Ear", "right fiducial":"Right Ear" ,"nasion fiducial":"Nasion"}

def txt_to_tag(txtname):
    '''Extract the fiducials from the brainsight text file'''
    
    # Check that the file has the proper coordinates.

    scanner = False
    ll = open(txtname).readlines()
    for l in ll:
        l = l.strip()
        if l[0] == '#':
            if l == "# Coordinate system: NIfTI:Scanner":
                scanner = True
    
    if not scanner:
        print("{} does not appear to be in NIfTI:Scanner coordinates".format(txtname))
    
    # Find the tags and write them in the correct format for AFNI.
    
    tags = {}
    for l in ll:
        l = l.strip()
        if l[0] != '#':
            l = l.split('\t')
            if l[0][-1]==' ':  #Remove trailing space that sometimes happens
                l[0]=l[0][:-1]
            if l[0] in fiducials:
                x, y, z = [float(l[i]) for i in [3, 4, 5]]
                if l[0] in alt_fid:
                    l[0] = alt_fid[l[0]]
                # the coordinates are LPI, convert them to RAI
                tags[l[0]] = "'{}' {} {} {}".format(l[0], -x, -y, z)
    return tags

def txt_to_tag_pd(txtname):
    '''Extract the fiducials from brainsight using pandas'''
    import pandas as pd
    dframe = pd.read_csv(txtname, skiprows=6, sep='\t')
    # fids_idx=dframe['Electrode Type'].isin(['LPA','Nasion','RPA'])
    # fids_dframe=dframe.loc[fids_idx]   
    lpa_idx = dframe[dframe['Electrode Type']=='LPA']
    rpa_idx = dframe[dframe['Electrode Type']=='RPA']
    nas_idx = dframe[dframe['Electrode Type']=='Nasion']
    
    if (0 < len(nas_idx) > 2) or (0 < len(rpa_idx) > 2) or (0 < len(lpa_idx) > 2):
         raise BaseException('''Too many entries in the brainsight file 
                             listed for LPA, RPA, Nasion''')
    tags={}
    for key, ser in zip(['Nasion','Right Ear','Left Ear'],[nas_idx, rpa_idx, lpa_idx]):
        x=ser['Loc. X'].values[0]
        y=ser['Loc. Y'].values[0]
        z=ser['Loc. Z'].values[0]
        tags[key]="'{}' {} {} {}".format(key, -x, -y, z)
    return tags


def tags_from_bsight_targetfile(fname, tag_template=['NAS','LPA','RPA']):
    '''
    If brainsight files are exported from the target menu, the headers are 
    inconsistent the electrodes.txt style export file.  This strips out the 
    non-matching column widths.

    Parameters
    ----------
    fname : TYPE
        DESCRIPTION.
    tag_template : TYPE, optional
        DESCRIPTION. The default is ['NAS','LPA','RPA'].

    Raises
    ------
    ValueError
        If start or stopping conditions in the target file can not be found 
        this will result in an error.

    Returns
    -------
    tags : dict
        {"NAS": float, "LPA": float, "RPA": float}

    '''
    import pandas as pd
    with open(fname) as f:
        tmp = f.readlines()
    
    #Identify start and end of target section
    idx=0; start_idx=0; end_idx=0
    for i in tmp:
        if i[0:8]=='# Sample':
            start_idx=idx
        if i[0:9]=='# Planned':
            end_idx=idx
        idx+=1
    
    if (start_idx==0) or (end_idx==0):
        raise ValueError('Cannot find the start and end of localizer section')
    
    #Format text and create list
    tmp_ = tmp[start_idx:end_idx]        
    output = []
    for i in tmp_:
        output.append(i.strip('\n').split('\t'))
    output[0]=['Sample Name']+output[0][1:]
    
    #Create dataframe and extract tags
    dframe = pd.DataFrame(output[1:], columns=output[0])
    tags ={}
    for tag in tag_template:
        tmp = dframe[dframe['Sample Name']==tag][['Loc. X','Loc. Y','Loc. Z']].values[0]
        tags[tag] = [float(i) for i in tmp]
    final_tags = {}
    for key in tags.keys():
        if key[0].upper()=='L':
            final_tags['Left Ear']=tags[key]
        if key[0].upper()=='R':
            final_tags['Right Ear']=tags[key]
        if key[0].upper()=='N':
            final_tags['Nasion'] = tags[key]
    assert ['Left Ear','Right Ear','Nasion'] in list(final_tags.keys())
    return final_tags


def write_tagfile(tags, out_fname=None):
    ''''''
    
    print("writing {}".format(out_fname))
    f = open(out_fname, 'w')
    for tag in fiducials0:
        print(tags[tag], file = f)
    f.close()    
    


def main():
    if len(sys.argv) != 2:
        print("""usage: {} file.txt
          Where file.txt is the saved electrode location output from Brainsight.""".format(sys.argv[0]))
        sys.exit(1)
    
    # Generate the name for the tag file
    txtname=sys.argv[1] 
    name, ext = os.path.splitext(txtname)
    if ext != ".txt":
        name = txtname
    tagname = "{}.tag".format(name)
    
    tags = txt_to_tag(txtname)
    write_tagfile(tags, tagname)
    print('Finished tag export')
    
if __name__=='__main__':
    main()
    
    

    

