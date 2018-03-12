#!/usr/bin/env python
import xml.etree.ElementTree as XET

class odsread:
    def __init__(self,filename):
        from zipfile import ZipFile
        ziparchive = ZipFile(filename, "r")
        self.xmldata = ziparchive.read("content.xml")
        ziparchive.close()
        self.xmltree=XET.fromstring(self.xmldata)

    def __tag2str(self,tag):
        return tag.split("}")[-1]

    def __findtag(self,tree,tag):
        for branch in list(tree):
            if self.__tag2str(branch.tag)==tag:
                return branch
            else:
                branch2=self.__findtag(branch,tag)
                if isinstance(branch2,XET.Element):
                    return branch2

    def __findtagall(self,treein,tagin):
        def taginteg(tree,tag):
            for branch in list(tree):
                if self.__tag2str(branch.tag)==tag:
                    branchall.append(branch)
                else:
                    taginteg(branch,tag)
        branchall=[]
        taginteg(treein,tagin)
        return branchall

    def __getattrib(self,tree,name):
        for attrib in tree.attrib:
            if "}"+name in attrib:
                return tree.get(attrib)
        return False

    def __findtreewithattrib(self,trees,attribname,attribvalue):
        for tree in trees:
            if self.__getattrib(tree,attribname)==attribvalue:
                return tree
        return False

    def parse(self,tablename):
        tables=self.__findtagall(self.xmltree,"table")
        table=self.__findtreewithattrib(tables,"name",tablename)
        rows=self.__findtagall(table,"table-row")
        self.values,self.hrefs=[],[] #value and link
        for row in rows:
            cells=self.__findtagall(row,"table-cell")
            self.values.append([])
            self.hrefs.append([])
            for cell in cells:
                if self.__getattrib(cell,"number-columns-repeated"):
                    repeat=int(self.__getattrib(cell,"number-columns-repeated"))
                else:repeat=1
                if repeat>500:repeat=1
                for i in range(repeat):
                    text=cell.itertext()
                    self.values[-1].append("".join(text))
                    hreftag=self.__findtag(cell,"a")
                    if hreftag!=None:
                        hrefkey = [k for k in hreftag.attrib.keys() if "href" in k][0]
                        self.hrefs[-1].append(hreftag.attrib[hrefkey])
                    else:
                        self.hrefs[-1].append(None)

    def getvalue(self,row,col):
        if row<len(self.values):
            if col<len(self.values[row]):
                return self.values[row][col]
        return False

    def getvalbyrow(self,row):
        if row<len(self.values):
            return self.values[row]
        return False

    def getvalbycol(self,col):
        vals=[]
        for rows in self.values:
            if col<len(rows):
                vals.append(rows[col])
            else:
                vals.append("")
        return vals

if __name__ == '__main__':
    bcmconstfile="/home/pzhu/work/run record/bcm calibration.ods"
    ods=odsread(bcmconstfile)
    ods.parse("bcm")
    for col in ods.getvalbycol(14):
        print(col)
